# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Typer application for the `liveaov` CLI.

Phase 0 surface:
  - `liveaov --version`
  - `liveaov plugins list [--type <pass_type>]`
  - `liveaov run-shot <folder> --passes <csv>` - run registered passes on a
    sequence folder and write sidecar EXRs

Phase 4 fleshes this out with discover / analyze / run / preflight / models.
"""

from __future__ import annotations

import os
import re
import signal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

import live_action_aov
from live_action_aov.core.cancel import CancelledError, CancelToken
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.output_layout import next_version_dir, plate_sibling_dir
from live_action_aov.core.pass_base import License
from live_action_aov.core.registry import get_registry
from live_action_aov.core.work_dir import set_work_root
from live_action_aov.io.readers.oiio_exr import OIIOExrReader

app = typer.Typer(
    name="liveaov",
    help="LiveActionAOV - VFX plate -> AOV sidecar preprocessor.",
    add_completion=False,
    no_args_is_help=True,
)

plugins_app = typer.Typer(help="Plugin discovery commands.")
app.add_typer(plugins_app, name="plugins")

console = Console()


# ---------------------------------------------------------------------------
# Top-level options
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"LiveActionAOV v{live_action_aov.__version__}")
        typer.echo("by Leonardo Paolini")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    offline: Annotated[
        bool,
        typer.Option(
            "--offline",
            help="Never contact Hugging Face - use only already-downloaded "
            "models (sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE). Run "
            "`liveaov prefetch` once online first.",
        ),
    ] = False,
    work_root: Annotated[
        Path | None,
        typer.Option(
            "--work-root",
            help="Folder for temporary frames exported from the plate (default: the "
            "system temp folder; sets LAOV_WORK_ROOT). Use a local disk, not a share.",
        ),
    ] = None,
) -> None:
    """Top-level options."""
    if work_root is not None:
        set_work_root(work_root)
    if offline:
        # Set before any pass imports huggingface_hub (loads are lazy). A
        # blip / outage can no longer abort a run whose weights are cached.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Commercial-safe default set a typical run uses - what `prefetch` warms
# when no explicit list is given, and what the portable build carries.
# Not in it on purpose: DSINE (non-commercial licence) and RVM (GPL-3.0 code:
# fetched from its repository at first use, never bundled).
_DEFAULT_PREFETCH = [
    "depth_anything_v2",
    "video_depth_anything",
    "moge3",
    "sam3_matte",
    "birefnet_refiner",
    "vitmatte_refiner",
    "flow",
    "flow_megaflow",
]

#: Registered passes that intentionally have no weights to prefetch (no
#: `_load_model` work) or are known stubs. Used by the coverage warning.
# chroma_key + screen_pull: pure maths, no weights to prefetch.
# marigold_v2_*: weights live in an EXTERNAL env (scripts/download_assets.py
# there), not in the HF cache this prefetch fills.
_PREFETCH_EXEMPT = {
    "cryptomatte",
    "matanyone2",
    "chroma_key",
    "screen_pull",
    "marigold_v2_albedo",
    "marigold_v2_depth_seethrough",
    "camera_pycolmap",
    "splat_gsplat",
    "ltx_alpha_refiner",  # weights live in the user's ComfyUI
    "rvm_refiner",  # GPL-3.0: fetched on demand (`prefetch --passes rvm_refiner`), never bundled
}


def _warn_uncovered_passes(covered: list[str]) -> None:
    """Warn when a registered, commercial-safe, weight-loading pass is not in
    the default prefetch set. This is the anti-regression guard: the default
    list drifts every time a pass is added (vitmatte_refiner in 0.7.0,
    video_depth_anything before it), and the gap only ever surfaced later as
    a failed --offline run - the exact failure prefetch exists to prevent.
    """
    from live_action_aov.core.pass_base import UtilityPass

    registry = get_registry()
    missing: list[str] = []
    for name in registry.list_passes():
        if name in covered or name in _PREFETCH_EXEMPT:
            continue
        cls = registry.get_pass(name)
        has_weights = any("_load_model" in c.__dict__ for c in cls.__mro__ if c is not UtilityPass)
        if has_weights and cls.declared_license().commercial_use:
            missing.append(name)
    if missing:
        console.print(
            "[yellow]Warning:[/yellow] these commercial-safe passes are NOT in "
            f"the default prefetch set and were not fetched: {', '.join(missing)}.\n"
            f"An --offline run using them will fail. Fetch them with:\n"
            f"  liveaov prefetch --passes {','.join(missing)}"
        )


@app.command("prefetch")
def prefetch(
    passes: Annotated[
        str | None,
        typer.Option(
            "--passes",
            "-p",
            help="Comma-separated pass names to prefetch. Default: the "
            "commercial-safe set (depth/normals/matte/soft-matte/flow).",
        ),
    ] = None,
    all_passes: Annotated[
        bool,
        typer.Option("--all", help="Prefetch every registered pass (incl. heavy backends)."),
    ] = False,
) -> None:
    """Download all model weights into the local cache ('dummy preload').

    Loads each pass's model once ON CPU (no GPU, no VRAM contention) purely
    to trigger every download - including secondary repos some passes pull
    (e.g. NormalCrafter / DepthCrafter -> stable-video-diffusion) and the
    torch.hub backends (RVM, DSINE). Run this once while online; afterwards
    you can run fully offline (`liveaov --offline ...` or HF_HUB_OFFLINE=1)
    on an unreliable link or a content machine.
    """
    # Force CPU before torch is imported anywhere: we only want the download
    # side effect, not a GPU load (keeps VRAM free, never fights ComfyUI).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    registry = get_registry()
    if all_passes:
        names = [
            n for n in registry.list_passes() if not getattr(registry.get_pass(n), "stub", False)
        ]
    elif passes:
        names = [p.strip() for p in passes.split(",") if p.strip()]
    else:
        names = list(_DEFAULT_PREFETCH)
        # Coverage guard: surface any commercial-safe weight-loading pass the
        # default set forgot, BEFORE the user trusts "all cached" and goes
        # offline. (Only for the default set - explicit --passes/--all are
        # the user's own selection.)
        _warn_uncovered_passes(names)

    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    for name in names:
        console.print(f"[cyan]Prefetching[/cyan] {name} ...")
        try:
            inst = registry.get_pass(name)()
            # `prefetch_weights` = download-only hook for passes whose model
            # cannot be built on CPU (MoGe-3's sparse refiner needs CUDA).
            load = getattr(inst, "prefetch_weights", None) or getattr(inst, "_load_model", None)
            if callable(load):
                load()  # triggers the download(s)
            unload = getattr(inst, "unload", None)
            if callable(unload):
                unload()
            ok.append(name)
            console.print(f"  [green]cached[/green] {name}")
        except Exception as e:
            failed.append((name, f"{type(e).__name__}: {e}"))
            console.print(f"  [yellow]skipped[/yellow] {name}: {type(e).__name__}: {e}")

    console.print(
        f"\n[bold]Prefetched {len(ok)}/{len(names)}[/bold]"
        + (f" - {len(failed)} failed (see above)." if failed else " - all cached.")
    )
    if ok and not failed:
        console.print("You can now run offline:  liveaov --offline run-shot <folder> ...")
    if failed:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# plugins list
# ---------------------------------------------------------------------------


@plugins_app.command("list")
def plugins_list(
    pass_type: Annotated[
        str | None,
        typer.Option("--type", help="Filter by pass type (geometric/motion/semantic/...)."),
    ] = None,
) -> None:
    """List registered passes discovered via entry points."""
    registry = get_registry()
    names = registry.list_by_type(pass_type) if pass_type else registry.list_passes()
    table = Table(title="Registered passes")
    table.add_column("name")
    table.add_column("type")
    table.add_column("license")
    table.add_column("commercial")
    for name in names:
        cls = registry.get_pass(name)
        try:
            lic: License = cls.declared_license()
            lic_str = lic.spdx
            commercial = "yes" if lic.commercial_use else "no"
        except Exception:
            lic_str = "?"
            commercial = "?"
        pt = getattr(cls, "pass_type", None)
        pt_str = str(pt.value) if pt is not None and hasattr(pt, "value") else str(pt or "?")
        table.add_row(name, pt_str, lic_str, commercial)
    console.print(table)


# ---------------------------------------------------------------------------
# run-shot
# ---------------------------------------------------------------------------


@app.command("run-shot")
def run_shot(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False, dir_okay=True)],
    passes: Annotated[
        str,
        typer.Option(
            "--passes",
            "-p",
            help="Comma-separated pass list. Semantic names `depth` and `normals` "
            "are rewritten to the --depth-backend / --normals-backend choice.",
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Output root: each run lands in <out>/<shot>/v###/. Default: "
            "next to the plate, <plate folder>_utility/v###/ (a new version every run).",
        ),
    ] = None,
    depth_backend: Annotated[
        str,
        typer.Option(
            "--depth-backend",
            help="Backend for the semantic `depth` pass. "
            "Default: depth_anything_v2 (Apache-2.0, commercial OK).",
        ),
    ] = "depth_anything_v2",
    normals_backend: Annotated[
        str,
        typer.Option(
            "--normals-backend",
            help="Backend for the semantic `normals` pass. Default: moge3 (MIT, "
            "sharp, stable). `dsine` is lighter but non-commercial.",
        ),
    ] = "moge3",
    matte_detector: Annotated[
        str,
        typer.Option(
            "--matte-detector",
            help="Detector for the semantic `matte` pass. Default: sam3_matte "
            "(SAM-License-1.0, commercial OK with military carve-out).",
        ),
    ] = "sam3_matte",
    refiner: Annotated[
        str,
        typer.Option(
            "--refiner",
            help="Refiner for the semantic `matte` pass. Default: rvm_refiner "
            "(GPL-3.0-only; its mattes are fine for paid work, the model is "
            "fetched from its own repository). `matanyone2` (NTU-S-Lab-1.0) is "
            "non-commercial.",
        ),
    ] = "rvm_refiner",
    allow_noncommercial: Annotated[
        bool,
        typer.Option(
            "--allow-noncommercial",
            help="Permit passes whose license disallows commercial use.",
        ),
    ] = False,
    display_transform: Annotated[
        bool,
        typer.Option(
            "--display-transform/--no-display-transform",
            help="Apply a clip-uniform display transform (auto-exposure + "
            "AgX + sRGB EOTF) to every frame before passes see it. "
            "Required for scene-referred plates (lin_rec709, ACEScg, "
            "ACES); off by default so sRGB plates pass through "
            "unchanged. Exposure is analysed once per clip so there "
            "is no temporal flicker.",
        ),
    ] = False,
    colorspace: Annotated[
        str | None,
        typer.Option(
            "--colorspace",
            help="Override the plate colourspace. Accepts `lin_rec709`, "
            "`acescg`, `srgb_display`, `rec709_display`, etc. Use this "
            "when the EXR's `oiio:ColorSpace` tag is wrong - e.g. MOV->EXR "
            "conversions that preserve display-referred values but tag "
            "them as linear. When unset the display transform reader "
            "trusts whatever the plate's metadata says.",
        ),
    ] = None,
    first: Annotated[
        int | None,
        typer.Option("--first", help="First frame to process (default: the plate's first)."),
    ] = None,
    last: Annotated[
        int | None,
        typer.Option("--last", help="Last frame to process (default: the plate's last)."),
    ] = None,
    crop: Annotated[
        str | None,
        typer.Option(
            "--crop",
            help="Process one box of the plate only: `x,y,w,h` in plate pixels, origin "
            "top-left. The EXRs keep the full plate as display window with the crop as "
            "data window, so Nuke places them. Camera and splat need the whole frame.",
        ),
    ] = None,
    naming: Annotated[
        str | None,
        typer.Option(
            "--naming",
            help="File naming template, e.g. `{shot}_{pass}_{version}.{frame}.{ext}`. "
            "Tokens: shot, plate, version, pass, frame, ext. Default "
            "`{shot}_{version}.{pass}.{frame}.{ext}`.",
        ),
    ] = None,
    proxy: Annotated[
        str | None,
        typer.Option(
            "--proxy",
            help="Run the models on a smaller copy of the plate: its long edge in "
            "pixels (1920, 2816 ...), or `auto` = what `liveaov doctor` proposes for "
            "this card. The EXRs are still written at plate size. Default: full plate.",
        ),
    ] = None,
    window: Annotated[
        str,
        typer.Option(
            "--window",
            help="Long clips: frames per window. `auto` (default) = one go when the "
            "clip fits the free RAM, else windows sized from it; a number forces "
            "that many frames; `off` = always one go.",
        ),
    ] = "auto",
) -> None:
    """Run passes on the EXR sequence in `folder` and write sidecar EXRs."""
    raw_names = [p.strip() for p in passes.split(",") if p.strip()]
    pass_names = _resolve_semantic_passes(
        raw_names,
        depth_backend=depth_backend,
        normals_backend=normals_backend,
        matte_detector=matte_detector,
        refiner=refiner,
    )
    registry = get_registry()

    # License gate - check before any I/O.
    blocked: list[tuple[str, str]] = []
    for name in pass_names:
        cls = registry.get_pass(name)
        lic = cls.declared_license()
        if not lic.commercial_use and not allow_noncommercial:
            blocked.append((name, lic.spdx))
    if blocked:
        lines = [f"  - {n} ({spdx})" for n, spdx in blocked]
        console.print(
            "[red]Refusing to run non-commercial passes without "
            "--allow-noncommercial:[/red]\n" + "\n".join(lines)
        )
        raise typer.Exit(code=2)

    # Discover the sequence - find one EXR, derive a `####` pattern.
    pattern, frame_range, resolution, pixel_aspect = _sniff_sequence(folder)
    plate_first, plate_last = frame_range
    work_first = plate_first if first is None else int(first)
    work_last = plate_last if last is None else int(last)
    if not (plate_first <= work_first <= work_last <= plate_last):
        console.print(
            f"[red]--first/--last {work_first}..{work_last} is outside the plate "
            f"{plate_first}..{plate_last}.[/red]"
        )
        raise typer.Exit(code=2)
    crop_box: tuple[int, int, int, int] | None = None
    if crop:
        try:
            parts = [int(p) for p in crop.replace(" ", "").split(",")]
            if len(parts) != 4:
                raise ValueError
            crop_box = (parts[0], parts[1], parts[2], parts[3])
        except ValueError:
            console.print("[red]--crop wants four integers: x,y,w,h (plate pixels).[/red]")
            raise typer.Exit(code=2) from None
    proxy_long_edge: int | None = None
    if proxy:
        if proxy.strip().lower() == "auto":
            from live_action_aov.core import hardware
            from live_action_aov.passes.matte import sam31_bridge

            suggestion = hardware.suggest(
                hardware.detect(),
                (crop_box[2], crop_box[3]) if crop_box else resolution,
                work_last - work_first + 1,
                sam31_installed=sam31_bridge.available({})[0],
            )
            proxy_long_edge = suggestion.proxy_long_edge
            console.print(f"[cyan]--proxy auto:[/cyan] {suggestion.one_line()}", highlight=False)
        else:
            try:
                proxy_long_edge = int(proxy)
                if proxy_long_edge < 64:
                    raise ValueError
            except ValueError:
                console.print("[red]--proxy wants `auto` or a long edge in pixels.[/red]")
                raise typer.Exit(code=2) from None
    try:
        shot = Shot(
            name=folder.name,
            folder=folder,
            proxy_long_edge=proxy_long_edge,
            window_frames=window,
            apply_display_transform=display_transform,
            colorspace=colorspace or "auto",
            sequence_pattern=pattern,
            frame_range=(work_first, work_last),
            resolution=resolution,
            pixel_aspect=pixel_aspect,
            passes_enabled=pass_names,
            output_dir=next_version_dir(
                out / folder.name if out is not None else plate_sibling_dir(folder)
            ),
            naming=naming or "",
            crop=crop_box,
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2) from None
    moge3 = _moge3_outputs(raw_names, depth_backend=depth_backend, normals_backend=normals_backend)
    job = Job(
        shot=shot,
        passes=[
            PassConfig(name=n, params=moge3) if n == "moge3" else PassConfig(name=n)
            for n in pass_names
        ],
    )

    console.print(
        f"[cyan]Running[/cyan] {shot.name} frames {frame_range[0]}..{frame_range[1]} "
        f"with passes: {pass_names}"
    )
    # Install a SIGINT handler that flips a CancelToken instead of
    # raising KeyboardInterrupt mid-inference. Two ergonomic wins:
    # (1) the executor finishes whatever frame is being written to disk
    # so the partial output is at least file-aligned; (2) we can print
    # a clean "Cancelled." line and exit 130 (Unix convention for
    # SIGINT) instead of dumping a Python traceback.
    cancel = CancelToken()
    previous_handler = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum: int, frame: object) -> None:
        if cancel.is_cancelled():
            # Second Ctrl+C - escalate to the previous handler (default
            # KeyboardInterrupt) so a stuck inference call still
            # responds to a forceful abort. We don't try to be clever
            # here; if the user double-tapped, they want out NOW.
            if callable(previous_handler):
                previous_handler(signum, frame)  # type: ignore[arg-type]
            else:
                raise KeyboardInterrupt
            return
        cancel.cancel("Cancelled by SIGINT")
        console.print("\n[yellow]Cancelling...[/yellow] (Ctrl+C again to force-quit)")

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        live_action_aov.run(job, cancel=cancel)
    except CancelledError:
        console.print("[yellow]Cancelled.[/yellow]")
        _print_run_report(shot)
        # 130 is the standard exit code for "terminated by SIGINT"
        # (128 + signal number). Lets shell scripts distinguish
        # user-cancelled runs from real failures.
        raise typer.Exit(code=130) from None
    except Exception as e:
        console.print(f"[red]FAILED:[/red] {e}")
        _print_run_report(shot)
        raise typer.Exit(code=1) from e
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    console.print(f"[green]Done.[/green] {shot.sidecars.get('comp') or shot.output_dir}")
    _print_run_report(shot)


def _print_run_report(shot: Shot) -> None:
    """The run report's one-line summary + its "needs a human" list."""
    from live_action_aov.core.run_report import load_report

    path = shot.sidecars.get("run_report")
    if path is None:
        return
    try:
        rep = load_report(Path(path))
    except Exception as e:  # the report is a convenience here
        console.print(f"[yellow]Run report unreadable:[/yellow] {e}", highlight=False)
        return
    console.print(f"[bold]Run report:[/bold] {rep.summary}", highlight=False)
    for n in rep.needs_human:
        where = f" (frames {n.frames[0]}-{n.frames[1]})" if n.frames else ""
        console.print(f"  [yellow]needs a human:[/yellow] {n.reason}{where}", highlight=False)
    console.print(f"  {path}", highlight=False)


# ---------------------------------------------------------------------------
# inspect - dump channels, metadata, and hero summary for a sidecar EXR
# ---------------------------------------------------------------------------


@app.command("inspect")
def inspect_cmd(
    sidecar: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to a sidecar EXR written by `liveaov run-shot`.",
        ),
    ],
    as_json: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit a machine-parseable JSON document instead of the "
            "human-readable text report.",
        ),
    ] = False,
) -> None:
    """Dump channels, metadata, and hero summary for a sidecar EXR.

    The first stop when verifying a pipeline run: confirms `mask.<concept>`
    + `matte.{r,g,b,a}` channels landed, the `liveaov/*` metadata
    block is stamped correctly, and shows which hero ended up in which
    slot. Pixel-level QC (edge quality, temporal flicker) still belongs in
    Nuke - this command is the plumbing check that comes before that.
    """
    from live_action_aov.cli import inspect as _inspect

    try:
        report = _inspect.build_report(sidecar)
    except Exception as e:
        # Bubble a clean one-liner rather than a traceback - `inspect` is
        # user-facing tooling, not a dev command.
        typer.echo(f"inspect: failed to read {sidecar}: {e}", err=True)
        raise typer.Exit(code=1) from e

    if as_json:
        typer.echo(_inspect.format_json_str(report))
    else:
        typer.echo(_inspect.format_text(report))


# ---------------------------------------------------------------------------
# doctor - what this machine can run, and the settings to use
# ---------------------------------------------------------------------------


@app.command("doctor")
def doctor(
    plate: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="A plate folder: the suggestion is then made for that shot "
            "(its size and length). Without it: for a 4K, 100-frame shot.",
        ),
    ] = None,
    comfy_url: Annotated[
        str,
        typer.Option("--comfy-url", help="The ComfyUI that runs LTX Alpha."),
    ] = "http://127.0.0.1:8188",
) -> None:
    """Check this machine and propose settings: the card and RAM, SAM 3.1,
    ComfyUI + the LTX models, gsplat, and the proxy size / detector /
    long-shot windows that fit."""
    import importlib.util

    from live_action_aov.core import hardware
    from live_action_aov.gui.cuda_check import cuda_state
    from live_action_aov.passes.matte import sam31_bridge
    from live_action_aov.passes.matte.ltx_alpha import comfy_check

    def line(ok: bool | None, what: str, detail: str) -> None:
        mark = {
            True: "[green]OK  [/green]",
            False: "[red]NO  [/red]",
            None: "[yellow]--  [/yellow]",
        }
        console.print(f"{mark[ok]} [bold]{what}[/bold]: {detail}", highlight=False)

    hw = hardware.detect()
    cuda = cuda_state()
    line(cuda.available, "GPU", hw.summary() if cuda.available else cuda.advisory)
    if hw.ram_free_gb is not None:
        line(None, "RAM free now", f"{hw.ram_free_gb:.0f} GB")
    sam31_ok, sam31_why = sam31_bridge.available({})
    line(
        sam31_ok or None,
        "SAM 3.1",
        sam31_why if sam31_ok else "not installed - `liveaov setup-sam31` builds it",
    )
    ltx_ok, ltx_msg = comfy_check({"comfy_url": comfy_url})
    line(
        ltx_ok or None,
        "LTX Alpha (ComfyUI)",
        f"ready at {comfy_url}" if ltx_ok else f"{ltx_msg} - mattes use ViTMatte until it is",
    )
    has_gsplat = importlib.util.find_spec("gsplat") is not None
    line(
        has_gsplat or None,
        "Gaussian splat background",
        "gsplat installed" if has_gsplat else "gsplat not installed (see the README)",
    )

    if plate is not None:
        try:
            _, frange, res, _ = _sniff_sequence(plate)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"doctor: {e}", err=True)
            raise typer.Exit(code=2) from e
        n = frange[1] - frange[0] + 1
        what = f"{plate.name} ({res[0]}x{res[1]}, {n} frames)"
    else:
        res, n = (4096, 2160), 100
        what = "a 4K, 100-frame shot"
    s = hardware.suggest(hw, res, n, sam31_installed=sam31_ok)
    console.print(f"\n[bold]Suggested for {what}[/bold] - {s.one_line()}", highlight=False)
    for note in s.notes:
        console.print(f"  - {note}", highlight=False)
    if s.tier != "none":
        flag = "" if s.proxy_long_edge is None else f" --proxy {s.proxy_long_edge}"
        console.print(
            f"  run-shot:{flag or ' (no --proxy)'}; the detector and the windows are automatic.",
            highlight=False,
        )


# ---------------------------------------------------------------------------
# setup-sam31 - build the SAM 3.1 environment
# ---------------------------------------------------------------------------


@app.command("setup-sam31")
def setup_sam31(
    folder: Annotated[
        Path | None,
        typer.Option(
            "--dir",
            help="Where to build it (about 8 GB). Default: a per-user folder "
            "that LiveActionAOV finds by itself.",
        ),
    ] = None,
    torch_backend: Annotated[
        str,
        typer.Option("--torch-backend", help="PyTorch build for the card (uv's name)."),
    ] = "cu128",
) -> None:
    """Build the SAM 3.1 environment: a Python 3.12 venv with PyTorch and
    Meta's sam3 package. Safe to run again: finished steps are skipped.
    The matte pass then uses SAM 3.1 by itself (`detector: auto`)."""
    from live_action_aov.core import sam31_setup

    try:
        python = sam31_setup.setup(folder, torch_backend=torch_backend, echo=typer.echo)
    except RuntimeError as e:
        typer.echo(f"setup-sam31: {e}", err=True)
        raise typer.Exit(code=1) from e
    sam31_setup.remember(python, echo=typer.echo)
    typer.echo(
        f"SAM 3.1 environment ready: {python}\n"
        "Last step, once: request access to facebook/sam3.1 on huggingface.co and log in "
        "(`hf auth login`). The checkpoint (~3.5 GB) downloads at the first run."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SEMANTIC_ALIASES: dict[str, str] = {
    "depth": "depth_backend",
    "normals": "normals_backend",
    # `matte` is a *compound* alias: it expands to detector + refiner so the
    # user gets both `mask.<concept>` channels and packed `matte.r/g/b/a`
    # heroes with one keyword. The DAG sorts them into the right order
    # (detector provides, refiner requires sam3_hard_masks + sam3_instances).
    "matte": "matte_detector+refiner",
}


def _resolve_semantic_passes(
    raw_names: list[str],
    *,
    depth_backend: str,
    normals_backend: str,
    matte_detector: str = "sam3_matte",
    refiner: str = "rvm_refiner",
) -> list[str]:
    """Rewrite semantic names (`depth`, `normals`, `matte`) to real pass names.

    - `depth`   -> depth_backend
    - `normals` -> normals_backend
    - `matte`   -> [matte_detector, refiner]   (compound: two passes)

    Users can also pass concrete backend names directly (e.g. `depthcrafter`
    or `sam3_matte`) and they flow through unchanged. Duplicates are
    collapsed in order.
    """
    scalar_mapping = {"depth": depth_backend, "normals": normals_backend}
    resolved: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        if name not in seen:
            resolved.append(name)
            seen.add(name)

    for name in raw_names:
        if name == "matte":
            # Compound expansion. Detector first so the DAG has a stable
            # starting point; the topological sort still corrects any
            # user-supplied ordering later.
            _add(matte_detector)
            _add(refiner)
            continue
        _add(scalar_mapping.get(name, name))
    return resolved


def _moge3_outputs(
    raw_names: list[str], *, depth_backend: str, normals_backend: str
) -> dict[str, bool]:
    """Which slots MoGe-3 fills: the ones resolved to it (or both when it is
    named directly), so it never overwrites another depth/normals backend."""
    if "moge3" in raw_names:
        return {"emit_depth": True, "emit_normals": True}
    return {
        "emit_depth": "depth" in raw_names and depth_backend == "moge3",
        "emit_normals": "normals" in raw_names and normals_backend == "moge3",
    }


def _sniff_sequence(folder: Path) -> tuple[str, tuple[int, int], tuple[int, int], float]:
    """Find an EXR sequence in `folder` and derive its pattern + metadata."""
    # Skip sidecar EXRs from previous runs - the sidecar writer injects
    # `.utility.`, `.hero.`, or `.mask.` before the frame token (see
    # `core.output_layout.sidecar_template`). sorted() puts them before the
    # plate alphabetically, which would otherwise make the sniffer pick
    # up e.g. 2-channel utility files and feed them into the display-
    # transform luma dot-product as a (H,W,2)@(3,) matmul crash.
    _SIDECAR_TOKENS = (".utility.", ".hero.", ".mask.")
    exrs = sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".exr"
        and not any(tok in p.name for tok in _SIDECAR_TOKENS)
    )
    if not exrs:
        raise FileNotFoundError(f"No .exr files found in {folder}")
    # Pick the first one, extract its frame-number component.
    sample = exrs[0].name
    m = re.search(r"(\d{3,})(?=[^\d]*\.exr$)", sample)
    if not m:
        pattern = sample  # treat as a single-frame literal
    else:
        width = len(m.group(1))
        pattern = sample[: m.start()] + ("#" * width) + sample[m.end() :]
    reader = OIIOExrReader(folder, pattern)
    return (
        pattern,
        reader.frame_range(),
        reader.resolution(),
        reader.pixel_aspect(),
    )


def main() -> None:
    """Console-script entry point (`liveaov`)."""
    # Must precede any huggingface_hub import (passes import it lazily).
    from live_action_aov.core.runtime_env import apply_hf_network_defaults

    apply_hf_network_defaults()
    app()


if __name__ == "__main__":
    main()
