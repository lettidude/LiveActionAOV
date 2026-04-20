# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Typer application for the `liveaov` CLI.

Phase 0 surface:
  - `liveaov --version`
  - `liveaov plugins list [--type <pass_type>]`
  - `liveaov run-shot <folder> --passes <csv>` — run registered passes on a
    sequence folder and write sidecar EXRs

Phase 4 fleshes this out with discover / analyze / run / preflight / models.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

import live_action_aov
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.pass_base import License
from live_action_aov.core.registry import get_registry
from live_action_aov.io.readers.oiio_exr import OIIOExrReader

app = typer.Typer(
    name="liveaov",
    help="LiveActionAOV — VFX plate → AOV sidecar preprocessor.",
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
) -> None:
    """Top-level options."""


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
            help="Backend for the semantic `normals` pass. Default: dsine (MIT).",
        ),
    ] = "dsine",
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
            help="Refiner for the semantic `matte` pass. `rvm_refiner` (MIT, "
            "commercial-safe default) or `matanyone2` (NTU-S-Lab-1.0, NC).",
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

    # License gate — check before any I/O.
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

    # Discover the sequence — find one EXR, derive a `####` pattern.
    pattern, frame_range, resolution, pixel_aspect = _sniff_sequence(folder)
    shot = Shot(
        name=folder.name,
        folder=folder,
        apply_display_transform=display_transform,
        sequence_pattern=pattern,
        frame_range=frame_range,
        resolution=resolution,
        pixel_aspect=pixel_aspect,
        passes_enabled=pass_names,
    )
    job = Job(
        shot=shot,
        passes=[PassConfig(name=n) for n in pass_names],
    )

    console.print(
        f"[cyan]Running[/cyan] {shot.name} frames {frame_range[0]}..{frame_range[1]} "
        f"with passes: {pass_names}"
    )
    try:
        live_action_aov.run(job)
    except Exception as e:
        console.print(f"[red]FAILED:[/red] {e}")
        raise typer.Exit(code=1) from e

    console.print(f"[green]Done.[/green] Sidecar example: {shot.sidecars.get('utility')}")


# ---------------------------------------------------------------------------
# inspect — dump channels, metadata, and hero summary for a sidecar EXR
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
    Nuke — this command is the plumbing check that comes before that.
    """
    from live_action_aov.cli import inspect as _inspect

    try:
        report = _inspect.build_report(sidecar)
    except Exception as e:
        # Bubble a clean one-liner rather than a traceback — `inspect` is
        # user-facing tooling, not a dev command.
        typer.echo(f"inspect: failed to read {sidecar}: {e}", err=True)
        raise typer.Exit(code=1) from e

    if as_json:
        typer.echo(_inspect.format_json_str(report))
    else:
        typer.echo(_inspect.format_text(report))


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

    - `depth`   → depth_backend
    - `normals` → normals_backend
    - `matte`   → [matte_detector, refiner]   (compound: two passes)

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


def _sniff_sequence(folder: Path) -> tuple[str, tuple[int, int], tuple[int, int], float]:
    """Find an EXR sequence in `folder` and derive its pattern + metadata."""
    # Skip sidecar EXRs from previous runs — the sidecar writer injects
    # `.utility.`, `.hero.`, or `.mask.` before the frame token (see
    # `executors.local._sidecar_pattern`). sorted() puts them before the
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
    app()


if __name__ == "__main__":
    main()
