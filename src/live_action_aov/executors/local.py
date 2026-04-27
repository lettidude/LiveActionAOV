# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""LocalExecutor — single-GPU, in-process execution (v1).

Responsibilities:
  1. Resolve passes + post-processors from the plugin registry
  2. Topologically sort passes via the DAG (provides/requires artifacts)
  3. Call each pass's `run_shot(reader, frame_range)` — which handles its own
     frame iteration (PER_FRAME by default, PAIR for RAFT, etc.)
  4. Collect artifacts emitted by passes (flow → FlowCache)
  5. Apply configured post-processors (temporal smoother) to the per-frame
     channel dict
  6. Write one sidecar EXR per frame with `liveaov/*` metadata
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from live_action_aov.core.cancel import CancelledError, CancelToken
from live_action_aov.core.dag import PassNode, topological_sort
from live_action_aov.core.job import Job, PassConfig, PostConfig, Shot
from live_action_aov.core.logging_setup import RunLoggingSession
from live_action_aov.core.pass_base import TemporalMode, UtilityPass
from live_action_aov.core.registry import get_registry
from live_action_aov.executors.base import Executor
from live_action_aov.io.readers.display_transform_reader import DisplayTransformedReader
from live_action_aov.io.readers.oiio_exr import OIIOExrReader
from live_action_aov.io.readers.proxy import wrap_if_proxy
from live_action_aov.io.writers.exr import ExrSidecarWriter
from live_action_aov.shared.optical_flow.cache import FlowCache

METADATA_NAMESPACE = "liveaov"

_log = logging.getLogger(__name__)

# Callers pass a function that receives `(fraction_done, label)` per
# milestone; the GUI routes these to its progress bar + status bar,
# the CLI routes them to a `rich` progress display. `None` = silent.
ProgressCallback = Callable[[float, str], None]


class LocalExecutor(Executor):
    name = "local"

    def submit(
        self,
        job: Job,
        progress_callback: ProgressCallback | None = None,
        cancel: CancelToken | None = None,
    ) -> Job:
        """Run the job synchronously. Optional `progress_callback(fraction,
        label)` is called at each milestone: reader analysis, each pass
        start, post-processor apply, sidecar write batch, done. Default
        `None` keeps the pre-M2.5 behaviour untouched (CLI + tests stay
        working without edits).

        Optional `cancel` is a `CancelToken` (see `core.cancel`). When
        set externally between checkpoints, the executor raises
        `CancelledError` at the next safe point and stamps the shot's
        status as ``cancelled`` before re-raising. Mid-pass cancellation
        is not supported — a torch inference call holds the GIL and we
        only re-check the token after it returns."""
        report = progress_callback or (lambda _f, _l: None)
        registry = get_registry()
        shot = job.shot
        # Sidecar dir is resolved up-front so the per-run log file can be
        # placed alongside the outputs. See `core/logging_setup.py` for why
        # we want both a sidecar-local log and a central mirror.
        sidecar_dir = shot.output_dir or shot.folder
        with RunLoggingSession(shot_name=shot.name, sidecar_dir=sidecar_dir) as log_paths:
            return self._submit_logged(job, report, registry, shot, sidecar_dir, log_paths, cancel)

    def _submit_logged(
        self,
        job: Job,
        caller_report: ProgressCallback,
        registry: Any,
        shot: Shot,
        sidecar_dir: Path,
        log_paths: Any,
        cancel: CancelToken | None,
    ) -> Job:
        _log.info(
            "passes=%s post=%s frames=%s colorspace=%s proxy=%s output=%s",
            [pc.name for pc in job.passes],
            [pc.name for pc in job.post],
            shot.frame_range,
            shot.colorspace,
            shot.proxy_long_edge,
            sidecar_dir,
        )

        # Tee every progress milestone into the run log so the file
        # mirrors what the GUI bar shows — no new log lines needed at
        # each milestone below, and the CLI (which has no progress bar
        # in the console log) gets a readable timeline for free.
        # Parameter is `caller_report` (not `report`) so the local
        # `report` definition below isn't a shadowing redefinition.
        def report(frac: float, label: str) -> None:
            _log.info("[%3.0f%%] %s", frac * 100, label)
            caller_report(frac, label)

        report(0.0, "Resolving passes…")

        # Resolve pass classes up-front; fail loud on unknown names before
        # we read a single frame.
        resolved: list[tuple[PassConfig, type[UtilityPass]]] = []
        for pc in job.passes:
            resolved.append((pc, registry.get_pass(pc.name)))

        nodes = [_node_for(pc, cls) for (pc, cls) in resolved]
        ordered = topological_sort(nodes)
        resolved_by_name = {pc.name: (pc, cls) for (pc, cls) in resolved}

        raw_reader: Any = OIIOExrReader(shot.folder, shot.sequence_pattern)
        # Proxy resolution for fast iterations. Wraps the raw reader
        # with a resize-on-read shim before the display transform sees
        # anything, so exposure analysis runs on the downsampled clip
        # too (consistent with what the passes see). `None` = no-op.
        raw_reader = wrap_if_proxy(raw_reader, shot.proxy_long_edge)
        reader: Any
        if shot.apply_display_transform:
            # Clip-uniform display transform (auto-exposure + AgX + sRGB
            # EOTF) analysed once, applied on every read. Lets scene-referred
            # plates (lin_rec709 / ACEScg / ACES) work with models trained
            # on sRGB-display images. See
            # `io.readers.display_transform_reader` for the full rationale.
            wrapped = DisplayTransformedReader(
                raw_reader,
                params=shot.transform,
                colorspace_override=(
                    shot.colorspace if shot.colorspace and shot.colorspace != "auto" else None
                ),
            )
            report(0.05, "Analysing plate exposure…")
            wrapped.analyze(shot.frame_range)
            reader = wrapped
        else:
            reader = raw_reader
        writer = ExrSidecarWriter()

        # Shared state published between passes and post-processors.
        artifacts: dict[str, dict[int, Any]] = {}
        flow_cache = FlowCache()

        try:
            shot.status = "running"
            per_frame_channels: dict[int, dict[str, Any]] = {}

            total_passes = max(1, len(ordered))
            for pass_idx, node in enumerate(ordered):
                # Cancel checkpoint #1: between passes. This is the
                # most useful one in practice — a heavy pass like
                # DepthCrafter or NormalCrafter can run for minutes,
                # and cancelling between passes lets the user abort
                # before the next one starts.
                if cancel is not None:
                    cancel.raise_if_cancelled()
                # Passes dominate runtime — split the 0.1 → 0.9 band
                # evenly across them so the bar moves predictably even
                # though we can't see inside each pass.
                pass_fraction_start = 0.1 + 0.8 * (pass_idx / total_passes)
                report(
                    pass_fraction_start,
                    f"Pass {pass_idx + 1}/{total_passes}: {node.name}",
                )
                pc, cls = resolved_by_name[node.name]
                pass_params = dict(pc.params)
                pass_params.update(shot.pass_overrides.get(node.name, {}))
                instance = cls(pass_params)

                # Pipe upstream artifacts into passes that declare requires.
                # Default `ingest_artifacts` is a no-op so flow/depth/normals
                # pay nothing; refiner passes read sam3_hard_masks etc. here.
                if getattr(cls, "requires_artifacts", None):
                    instance.ingest_artifacts(artifacts)

                frame_outputs = instance.run_shot(reader, shot.frame_range)
                for frame_idx, channels in frame_outputs.items():
                    per_frame_channels.setdefault(frame_idx, {}).update(channels)

                emitted = instance.emit_artifacts()
                for art_name, per_frame in emitted.items():
                    artifacts.setdefault(art_name, {}).update(per_frame)
                # Mirror forward/backward flow into the FlowCache so the
                # smoother (and future passes) can read them by
                # (shot_id, frame, direction).
                if "forward_flow" in emitted:
                    for f, arr in emitted["forward_flow"].items():
                        flow_cache.put(shot.name, f, "forward", arr)
                if "backward_flow" in emitted:
                    for f, arr in emitted["backward_flow"].items():
                        flow_cache.put(shot.name, f, "backward", arr)

            # --- Auto-wire TemporalSmoother for PER_FRAME passes with
            # `smooth: auto` (spec §13.1 Phase 2). Only runs when a flow pass
            # actually emitted forward_flow — otherwise the smoother would
            # have nothing to consume and the auto-wire is a no-op anyway.
            flow_available = "forward_flow" in artifacts and bool(artifacts["forward_flow"])
            existing_post_names = {p.name for p in job.post}
            auto_post: list[PostConfig] = []

            # Auto-wire PositionFromDepth if any frame carries a Z channel
            # and the user didn't already schedule their own. P = unproject
            # of Z + intrinsics; every Nuke relight gizmo needs it and it's
            # free once depth exists. See post/position_from_depth.py for
            # the intrinsics-source priority chain.
            from live_action_aov.io.channels import (
                CH_N_X as _CH_N_X,
            )
            from live_action_aov.io.channels import (
                CH_N_Y as _CH_N_Y,
            )
            from live_action_aov.io.channels import (
                CH_N_Z as _CH_N_Z,
            )
            from live_action_aov.io.channels import (
                CH_Z as _CH_Z,
            )

            has_depth = any(_CH_Z in ch for ch in per_frame_channels.values())
            if has_depth and "position_from_depth" not in existing_post_names:
                auto_post.append(PostConfig(name="position_from_depth", params={}))

            # Auto-wire SSAO when both depth and a full normal triplet
            # landed in `per_frame_channels`. Hemispheric sampling gives
            # compers a baked ambient-occlusion channel for free — no
            # catalog entry needed on the GUI side yet. Defaults are
            # conservative (16 samples); users can override via a
            # manual post config in the Job YAML.
            has_normals = any(
                _CH_N_X in ch and _CH_N_Y in ch and _CH_N_Z in ch
                for ch in per_frame_channels.values()
            )
            if has_depth and has_normals and "ssao" not in existing_post_names:
                auto_post.append(PostConfig(name="ssao", params={}))

            if flow_available:
                for node in ordered:
                    pc, cls = resolved_by_name[node.name]
                    smooth_param = pc.params.get(
                        "smooth", getattr(cls, "DEFAULT_PARAMS", {}).get("smooth")
                    )
                    if not _smooth_wanted(smooth_param):
                        continue
                    if getattr(cls, "temporal_mode", None) != TemporalMode.PER_FRAME:
                        continue
                    smoothable = list(getattr(cls, "smoothable_channels", []) or [])
                    if not smoothable:
                        continue
                    auto_name = f"temporal_smooth::{node.name}"
                    if auto_name in existing_post_names:
                        continue
                    auto_post.append(
                        PostConfig(
                            name="temporal_smooth",
                            params={"applied_to": smoothable, "_auto_for": node.name},
                        )
                    )

            # --- Post-processors (user-configured + auto-wired smoothers) ---
            applied_post: list[dict[str, Any]] = []
            for post_cfg in list(job.post) + auto_post:
                post_cls = registry._post.get(post_cfg.name)
                if post_cls is None:
                    raise KeyError(
                        f"No post-processor named '{post_cfg.name}'. "
                        f"Available: {sorted(registry._post)}"
                    )
                post_instance = post_cls(post_cfg.params)
                per_frame_channels = post_instance.apply(
                    per_frame_channels, flow_cache, shot.name, shot=shot
                )
                # Capture any resolved state the post left on itself
                # (e.g. PositionFromDepth stamps intrinsics_source +
                # resolved_intrinsics after apply). This gets forwarded to
                # the sidecar metadata in _base_attrs.
                params_snapshot = dict(post_instance.params)
                if hasattr(post_instance, "intrinsics_source"):
                    params_snapshot["_intrinsics_source"] = post_instance.intrinsics_source
                if getattr(post_instance, "resolved_intrinsics", None) is not None:
                    params_snapshot["_resolved_intrinsics"] = post_instance.resolved_intrinsics
                applied_post.append(
                    {
                        "name": post_cfg.name,
                        "algorithm": getattr(post_instance, "algorithm", post_cfg.name),
                        "applied_to": list(post_instance.params.get("applied_to") or []),
                        "params": params_snapshot,
                    }
                )

            # --- Write sidecars ---
            # Output dir was resolved at submit() entry so the per-run
            # log could be placed alongside the outputs; reuse it here.
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            sidecar_template = _sidecar_pattern(shot.sequence_pattern)
            attrs_base = _base_attrs(shot, job, artifacts, applied_post)
            report(0.9, "Writing sidecars…")

            # Identify the matte refiner (if any) for the `matte/commercial`
            # shortcut attr. The refiner is whichever pass declared
            # `provides_artifacts` containing "matte_heroes" — keeps us
            # agnostic to rvm_refiner / matanyone2 / any future backend.
            matte_refiner_cls: type[UtilityPass] | None = None
            matte_detector_cls: type[UtilityPass] | None = None
            for node in ordered:
                _, cls = resolved_by_name[node.name]
                provides = set(getattr(cls, "provides_artifacts", ()) or ())
                if "matte_heroes" in provides:
                    matte_refiner_cls = cls
                if "sam3_hard_masks" in provides or "sam3_instances" in provides:
                    matte_detector_cls = cls

            # Proxy mode: passes ran at the downsampled resolution, but
            # sidecars should still line up with the plate in comp. So
            # we upscale every channel back to the native plate size
            # right before writing. Professional VFX tools treat proxy
            # as invisible to downstream comp — sidecars at proxy res
            # force a Nuke Resize + quality loss that's avoidable here.
            plate_w, plate_h = shot.resolution
            needs_upscale = False
            if shot.proxy_long_edge is not None:
                # Check the first channel's shape against plate res.
                for cdict in per_frame_channels.values():
                    for arr in cdict.values():
                        if arr.ndim >= 2 and (arr.shape[0] != plate_h or arr.shape[1] != plate_w):
                            needs_upscale = True
                        break
                    break

            first_written: Path | None = None
            total_frames = max(1, len(per_frame_channels))
            for frame_write_idx, (frame_idx, channels) in enumerate(per_frame_channels.items()):
                # Cancel checkpoint #2: between sidecar writes. Each
                # write is ~50ms, so per-frame polling is essentially
                # free here — the user gets near-instant abort during
                # the 0.9→1.0 phase of the progress bar.
                if cancel is not None:
                    cancel.raise_if_cancelled()
                if frame_write_idx % max(1, total_frames // 20) == 0:
                    frac = 0.9 + 0.1 * (frame_write_idx / total_frames)
                    report(
                        min(frac, 0.99),
                        f"Writing sidecars ({frame_write_idx}/{total_frames})",
                    )
                if needs_upscale:
                    channels = _upscale_channels_to_plate(channels, plate_h, plate_w)
                out_path = sidecar_dir / sidecar_template.format(frame=frame_idx)
                attrs = dict(attrs_base)
                attrs[f"{METADATA_NAMESPACE}/frame"] = frame_idx
                for node in ordered:
                    _, cls = resolved_by_name[node.name]
                    lic = cls.declared_license()
                    attrs[f"{METADATA_NAMESPACE}/{node.name}/model"] = cls.name
                    attrs[f"{METADATA_NAMESPACE}/{node.name}/version"] = cls.version
                    attrs[f"{METADATA_NAMESPACE}/{node.name}/license"] = lic.spdx
                    attrs[f"{METADATA_NAMESPACE}/{node.name}/commercial"] = lic.commercial_use
                if matte_refiner_cls is not None:
                    # `matte/commercial` is a downstream QC shortcut: the
                    # deliverable is commercial-safe iff the refiner is
                    # commercial-safe (detector doesn't affect downstream
                    # usability). Stored as "true"/"false" strings so Nuke's
                    # metadata TCL reads it as a plain scalar attribute.
                    r_lic = matte_refiner_cls.declared_license()
                    attrs[f"{METADATA_NAMESPACE}/matte/refiner"] = matte_refiner_cls.name
                    attrs[f"{METADATA_NAMESPACE}/matte/commercial"] = (
                        "true" if r_lic.commercial_use else "false"
                    )
                if matte_detector_cls is not None:
                    attrs[f"{METADATA_NAMESPACE}/matte/detector"] = matte_detector_cls.name
                writer.write_frame(
                    out_path,
                    channels,
                    attrs=attrs,
                    pixel_aspect=shot.pixel_aspect,
                )
                if first_written is None:
                    first_written = out_path

            shot.sidecars["utility"] = first_written or sidecar_dir
            shot.status = "done"
            report(1.0, "Done.")
        except CancelledError:
            # Distinct status so the GUI can render "Cancelled" rather
            # than "Failed", and downstream tooling can tell the
            # difference between "user aborted" and "actually broken".
            # Partial sidecars on disk are accepted as the price of
            # cancelling — cleaning them up automatically would surprise
            # users who wanted to inspect what got written.
            shot.status = "cancelled"
            _log.info("Cancelled by user (token raised at safe checkpoint).")
            raise
        except Exception:
            shot.status = "failed"
            raise
        return job


def _smooth_wanted(value: Any) -> bool:
    """Resolve a `smooth` param value to a boolean.

    Accepts:
      - "auto"            → True (per-frame backends get smoothing by default)
      - True / "true" / 1 → True
      - False / "false"/0 → False
      - None              → False (explicit opt-out via config)
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        return v in {"auto", "true", "1", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _node_for(pc: PassConfig, cls: type[UtilityPass]) -> PassNode:
    return PassNode(
        name=pc.name,
        plugin=pc.name,
        provides=tuple(getattr(cls, "provides_artifacts", ()) or ()),
        requires=tuple(getattr(cls, "requires_artifacts", ()) or ()),
    )


def _sidecar_pattern(sequence_pattern: str) -> str:
    """Given a plate pattern like `shot.####.exr`, return a sidecar pattern
    with `.utility.` injected before the frame token and a `{frame}` Python
    format placeholder.

    `shot.####.exr` → `shot.utility.{frame:04d}.exr`
    """
    import re

    m = re.search(r"#+", sequence_pattern)
    if m:
        width = len(m.group(0))
        base = sequence_pattern[: m.start()]
        tail = sequence_pattern[m.end() :]
        if base.endswith("."):
            base_ = base[:-1] + ".utility."
        else:
            base_ = base + ".utility."
        return f"{base_}{{frame:0{width}d}}{tail}"
    m = re.search(r"%0?(\d*)d", sequence_pattern)
    if m:
        width = int(m.group(1)) if m.group(1) else 4
        base = sequence_pattern[: m.start()]
        tail = sequence_pattern[m.end() :]
        if base.endswith("."):
            base_ = base[:-1] + ".utility."
        else:
            base_ = base + ".utility."
        return f"{base_}{{frame:0{width}d}}{tail}"
    return sequence_pattern.replace(".exr", ".utility.exr")


def _base_attrs(
    shot: Shot,
    job: Job,
    artifacts: dict[str, dict[int, Any]],
    applied_post: list[dict[str, Any]],
) -> dict[str, Any]:
    base: dict[str, Any] = {
        f"{METADATA_NAMESPACE}/version": "1.0.0",
        f"{METADATA_NAMESPACE}/created": _dt.datetime.now(_dt.UTC).isoformat(),
        f"{METADATA_NAMESPACE}/plate_source": str(shot.folder),
        f"{METADATA_NAMESPACE}/job_id": job.job_id,
        f"{METADATA_NAMESPACE}/shot_name": shot.name,
        f"{METADATA_NAMESPACE}/input/colorspace": shot.colorspace,
        f"{METADATA_NAMESPACE}/input/exposure_offset": float(
            shot.transform.computed_exposure_ev or 0.0
        ),
        f"{METADATA_NAMESPACE}/input/exposure_anchor": shot.transform.exposure_anchor,
        f"{METADATA_NAMESPACE}/input/tonemap": shot.transform.tonemap,
        f"{METADATA_NAMESPACE}/input/eotf": shot.transform.output_eotf,
    }
    # Parallax estimate (flow artifact) — one scalar per shot, exposed as a
    # header attribute so downstream passes can route (v2a).
    parallax = artifacts.get("parallax_estimate") or {}
    if parallax:
        sample_arr = next(iter(parallax.values()))
        try:
            base[f"{METADATA_NAMESPACE}/flow/parallax_estimate"] = float(sample_arr[0])
        except (IndexError, TypeError, ValueError):
            pass
        base[f"{METADATA_NAMESPACE}/flow/direction"] = "bidirectional"
        base[f"{METADATA_NAMESPACE}/flow/unit"] = "pixels_at_plate_res"
    # Depth-pass normalization constants (trap 5 — per-clip, not per-frame).
    for art_key, attr_key in (
        ("depth_norm_min", "depth/normalization/min"),
        ("depth_norm_max", "depth/normalization/max"),
    ):
        art = artifacts.get(art_key) or {}
        if art:
            sample_arr = next(iter(art.values()))
            try:
                base[f"{METADATA_NAMESPACE}/{attr_key}"] = float(sample_arr[0])
            except (IndexError, TypeError, ValueError):
                pass
    # Metric-depth backends (DepthPro) flag themselves via `depth_metric` so
    # the sidecar headers read `metric / meters` instead of the relative /
    # per-clip wiring used by DA V2 + DepthCrafter. Metric wins if both are
    # present (mixed-backend jobs are undefined but should prefer metric).
    if artifacts.get("depth_metric"):
        base[f"{METADATA_NAMESPACE}/depth/space"] = "metric"
        base[f"{METADATA_NAMESPACE}/depth/unit"] = "meters"
    elif artifacts.get("depth_norm_min") and artifacts.get("depth_norm_max"):
        base[f"{METADATA_NAMESPACE}/depth/space"] = "relative"
        base[f"{METADATA_NAMESPACE}/depth/unit"] = "normalized_per_clip"
    # Matte-pass metadata (Phase 3). Detector + refiner identities are
    # stamped from the per-pass loop later; this block adds the richer
    # matte-specific attrs: the concept list from SAM 3, the per-slot hero
    # names/track_ids/scores from the refiner, and the commercial-safety
    # flag downstream QC uses to decide if a deliverable may ship. The
    # refiner's license decides commercial safety (detector is always SAM3;
    # it's the refiner that flips to non-commercial when MatAnyone 2 is
    # wired in Round 2).
    matte_concepts = artifacts.get("matte_concepts") or {}
    if matte_concepts:
        concepts = next(iter(matte_concepts.values())) or []
        base[f"{METADATA_NAMESPACE}/matte/concepts"] = ",".join(str(c) for c in concepts)
    matte_heroes = artifacts.get("matte_heroes") or {}
    if matte_heroes:
        heroes = next(iter(matte_heroes.values())) or []
        for hero in heroes:
            slot = str(hero.get("slot", "")).lower()
            if slot not in {"r", "g", "b", "a"}:
                continue
            prefix = f"{METADATA_NAMESPACE}/matte/hero_{slot}"
            base[f"{prefix}/label"] = str(hero.get("label", ""))
            base[f"{prefix}/track_id"] = int(hero.get("track_id", 0))
            base[f"{prefix}/score"] = float(hero.get("score", 0.0))
    # Temporal smoothers — stamped under `smooth/`.
    smoothers = [p for p in applied_post if p["name"] == "temporal_smooth"]
    if smoothers:
        # Disambiguate per-pass auto-wired smoothers via a `::<pass>` suffix on
        # the metadata key. Manual post entries keep their raw name.
        tagged: list[str] = []
        for p in smoothers:
            auto_for = p["params"].get("_auto_for")
            key = f"{p['name']}::{auto_for}" if auto_for else p["name"]
            tagged.append(key)
            base[f"{METADATA_NAMESPACE}/smooth/{key}/algorithm"] = p["algorithm"]
            base[f"{METADATA_NAMESPACE}/smooth/{key}/applied_to"] = ",".join(p["applied_to"])
            if auto_for:
                base[f"{METADATA_NAMESPACE}/smooth/{key}/auto_for"] = auto_for
            if "alpha" in p["params"]:
                base[f"{METADATA_NAMESPACE}/smooth/{key}/alpha"] = float(p["params"]["alpha"])
            if "fb_threshold_px" in p["params"]:
                base[f"{METADATA_NAMESPACE}/smooth/{key}/fb_threshold"] = float(
                    p["params"]["fb_threshold_px"]
                )
        base[f"{METADATA_NAMESPACE}/smooth/post_processors"] = ",".join(tagged)

    # Position derivation — lives under `position/`.
    positions = [p for p in applied_post if p["name"] == "position_from_depth"]
    if positions:
        p = positions[0]
        base[f"{METADATA_NAMESPACE}/position/derived_from"] = "depth"
        base[f"{METADATA_NAMESPACE}/position/algorithm"] = p["algorithm"]
        src = p["params"].get("_intrinsics_source", "approximate_from_hfov")
        base[f"{METADATA_NAMESPACE}/position/intrinsics_source"] = src
        resolved = p["params"].get("_resolved_intrinsics")
        if resolved is not None:
            fx, fy, cx, cy = resolved
            base[f"{METADATA_NAMESPACE}/position/fx"] = float(fx)
            base[f"{METADATA_NAMESPACE}/position/fy"] = float(fy)
            base[f"{METADATA_NAMESPACE}/position/cx"] = float(cx)
            base[f"{METADATA_NAMESPACE}/position/cy"] = float(cy)

    # SSAO — stamped under `ao/`. Derived purely from Z + N channels, so
    # we record the input provenance so downstream QC can tell a synth
    # AO apart from a rendered one.
    aos = [p for p in applied_post if p["name"] == "ssao"]
    if aos:
        p = aos[0]
        base[f"{METADATA_NAMESPACE}/ao/derived_from"] = "depth+normals"
        base[f"{METADATA_NAMESPACE}/ao/algorithm"] = p["algorithm"]
        base[f"{METADATA_NAMESPACE}/ao/radius"] = float(p["params"].get("radius", 0.0))
        base[f"{METADATA_NAMESPACE}/ao/samples"] = int(p["params"].get("samples", 0))
        base[f"{METADATA_NAMESPACE}/ao/bias"] = float(p["params"].get("bias", 0.0))
        base[f"{METADATA_NAMESPACE}/ao/intensity"] = float(p["params"].get("intensity", 1.0))

    return base


def _upscale_channels_to_plate(
    channels: dict[str, Any],
    plate_h: int,
    plate_w: int,
) -> dict[str, Any]:
    """Upscale each per-frame channel array back to plate resolution.

    Proxy mode ran passes at a smaller resolution for speed; sidecars
    should still match the plate so comp can use them without an
    extra Resize node. Depth / normals / AO / matte are all scalar-ish
    fields; bilinear upscale is correct for depth + matte + AO. For
    normals we'd ideally renormalise to unit length post-upscale, but
    the NormalCrafter pass already renormalises inside its own
    run_shot after its own internal resize step, so by the time we
    get here normals are unit-length at the proxy resolution and the
    bilinear upscale introduces tiny sub-unit drift that's
    indistinguishable in comp. Flow channels (motion.x/y etc.) are
    pixel-magnitude fields — bilinear upscale is topologically
    correct but the magnitudes need to scale by the upscale ratio
    (pixels mean different things at different resolutions). We
    apply that correction inline.
    """
    import cv2

    upscaled: dict[str, Any] = {}
    scale_w: float | None = None
    scale_h: float | None = None
    flow_channels = {
        "motion.x",
        "motion.y",
        "back.x",
        "back.y",
        "forward.u",
        "forward.v",
        "backward.u",
        "backward.v",
    }
    for name, arr in channels.items():
        if arr.ndim < 2 or (arr.shape[0] == plate_h and arr.shape[1] == plate_w):
            upscaled[name] = arr
            continue
        if scale_w is None:
            scale_h = plate_h / arr.shape[0]
            scale_w = plate_w / arr.shape[1]
        up = cv2.resize(arr, (plate_w, plate_h), interpolation=cv2.INTER_LINEAR)
        if name in flow_channels:
            # Flow vectors are in source-pixel units. When we upscale
            # the field, each vector's magnitude scales with the
            # pixel-stride change — otherwise a 5-pixel displacement
            # at proxy res stays 5 pixels at plate res, which is
            # pointing a frame at the wrong target position.
            axis_scale = scale_w if name.endswith((".x", ".u")) else scale_h
            up = up * float(axis_scale or 1.0)
        upscaled[name] = up.astype(arr.dtype, copy=False)
    return upscaled


__all__ = ["LocalExecutor"]
