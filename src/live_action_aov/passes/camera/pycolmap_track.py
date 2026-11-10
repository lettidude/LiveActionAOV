# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Classical SfM camera track via pycolmap (BSD-3-Clause, commercial-safe).

pycolmap ships Windows/Linux/macOS wheels from PyPI — no C++ build, no CUDA
toolkit, no conda env; SIFT runs on CPU. That is why it is the v2a backend
(see the camera-backend decision notes): it installs everywhere.

Pipeline:
  1. Write each display-referred frame as a contrast-normalised (CLAHE)
     greyscale PNG at the solve resolution. CLAHE matters: on a dark,
     low-texture interior (ACTIONVFX PF0081) plain SIFT registered 2/32
     frames; with CLAHE + a lower peak threshold, 32/32 at <1 px.
  2. Optional exclusion masks: when a SAM 3 matte pass ran in the same job,
     every tracked object (people, cars, ... = things that move) is
     dilated and masked OUT of feature extraction — the matchmove
     "garbage matte" step, for free.
  3. SIFT extraction -> sequential matching (video) -> incremental (or
     global) mapping; the largest reconstruction is the solve.
  4. COLMAP's OpenCV cameras -> Y-up OpenGL world_from_camera, Nuke ZXY
     Eulers; intrinsics rescaled from the solve resolution to the plate.
  5. Sidecars in `<output>/camera/`: `<shot>.camera.json` (record),
     `<shot>.camera.nk` (Nuke Camera2), `<shot>.points.ply` (sparse cloud),
     `<shot>_colmap/` (COLMAP sparse model — what Gaussian-splat trainers
     and other SfM tools read).

Honest limits:
- Scale is arbitrary (no metric recovery).
- Default model is SIMPLE_PINHOLE (one focal, centred principal point);
  SIMPLE_RADIAL also solves k1, recorded in the JSON but NOT applied to
  the Nuke camera — undistort the plate if k1 matters.
- One focal for the whole clip (no zooms).
- Needs parallax: a locked-off or pure-pan shot cannot be triangulated.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from live_action_aov.core.output_layout import CAMERA_DIR, file_name, run_stem, sub_dir
from live_action_aov.core.pass_base import (
    License,
    PassType,
    SidecarSpec,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.core.work_dir import make_work_dir
from live_action_aov.io.camera_schema import (
    BackendInfo,
    CameraSolve,
    Conventions,
    Extrinsic,
    Intrinsics,
    Quality,
    ScenePoint,
    ShotRef,
)
from live_action_aov.io.writers.nuke_script import render_nuke_camera_script
from live_action_aov.io.writers.usd import write_usd

_log = logging.getLogger(__name__)

# OpenCV (+X right, +Y down, +Z forward) -> OpenGL (+X right, +Y up, +Z back):
# a 180 deg rotation about X (det = +1).
_OPENCV_TO_OPENGL_FLIP = np.diag([1.0, -1.0, -1.0])

_SUPPORTED_MODELS = ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL")


class PyColmapCameraPass(UtilityPass):
    """Classical SfM camera solve via pycolmap (VIDEO_CLIP, drives its own
    pipeline in `run_shot`; produces sidecars, no EXR channels)."""

    name = "camera_pycolmap"
    version = "0.2.0"
    license = License(
        spdx="BSD-3-Clause",
        commercial_use=True,
        commercial_tool_resale=True,
        notes="pycolmap / COLMAP are BSD-3-Clause (SIFT patent expired 2020).",
    )
    pass_type = PassType.CAMERA
    temporal_mode = TemporalMode.VIDEO_CLIP
    needs_full_frame = True
    # SIFT wants display-referred contrast; scene-linear would crush shadows.
    input_colorspace = "srgb_display"

    produces_channels: list = []
    produces_sidecars = [
        SidecarSpec(name="camera", format="json"),
        SidecarSpec(name="camera_nk", format="nk"),
        SidecarSpec(name="camera_points", format="ply"),
    ]
    # Soft dep: SAM 3 masks exclude moving objects when SAM 3 is in the job.
    optional_artifacts = ["sam3_hard_masks", "forward_flow", "backward_flow"]
    # The solve itself, for downstream passes (Gaussian splat).
    provides_artifacts = ["camera_solve"]
    # No parallax (locked-off, pure pan) is a property of the shot, not a bug:
    # the run keeps its mattes / depth when the solve fails.
    footage_dependent = True

    DEFAULT_PARAMS: dict[str, Any] = {
        "matcher": "sequential",  # "sequential" (video) | "exhaustive" (short / unordered)
        "sequential_overlap": 10,
        "mapper": "incremental",  # "incremental" | "global" (GLOMAP-style, pycolmap 4)
        # Fewer solved frames than this = no camera (the run keeps the rest).
        "min_registered_fraction": 0.5,
        # The incremental mapper's initial pair is random: on the A65 car the
        # same plate solved 100/100 or 3/100 half the time. Each attempt uses
        # a fixed seed and re-runs only the mapping.
        "mapping_attempts": 4,
        "camera_model": "SIMPLE_PINHOLE",
        # Long edge of the images SIFT sees. PF0081 (2048 wide): 2048 ->
        # 0.86 px / 62 s per 32 frames, 1024 -> 0.6 px@1K / 21 s. Handheld
        # tests: 1920 gives the same cameras as 2816 at a third of SIFT's RAM.
        # Above the proxy size the frames are read from the full-res plate.
        "max_image_size": 1920,
        # SIFT extraction threads. 0 = as many as fit in free RAM (each
        # thread holds ~0.25 GB per megapixel; all 32 threads at 2816 px
        # took +42 GB for ~10 s and could crash a long shot).
        "sift_num_threads": 0,
        "sift_max_features": 8192,
        # COLMAP default 0.0067 starves dark / low-contrast plates.
        "sift_peak_threshold": 0.002,
        "contrast_normalize": True,  # CLAHE before SIFT
        # SAM 3 exclusion: [] = every tracked object; or a list of labels.
        "mask_objects": True,
        "mask_labels": [],
        "mask_dilate_frac": 0.01,  # of the long edge, per side
        # Nuke film back (mm). Focal in mm is derived from this; the
        # vertical aperture follows the plate aspect.
        "haperture_mm": 24.892,
        # No parallax (tripod pan / tilt, or a locked-off camera): COLMAP's
        # mapper gives up, so solve rotation only from its verified matches
        # (passes/camera/nodal.py). Accepted only if it fits that well.
        "nodal_fallback": True,
        "nodal_max_error_px": 1.5,  # median, at solve resolution
        # Image motion below this fraction of the long edge cannot measure
        # the focal: use the lens metadata (ARRI FocalLength), else a guess.
        "nodal_min_motion": 0.03,
        # Neither fits (fast handheld action: COLMAP registers a handful of
        # frames, the camera translates): depth-anchored PnP
        # (passes/camera/pnp.py) - MoGe-3 depth of the middle frame + its
        # corners tracked through the shot. Judged on held-out points
        # (every `pnp_holdout_every`-th one never enters the solve): GOOD /
        # WEAK limits are fractions of the image width. A FAILED camera is
        # still written when `pnp_keep_failed`, labelled "do not use".
        # The same limits judge COLMAP's camera (`_referee_sfm`): KLT corners
        # tracked through the shot, held out frame by frame against its poses.
        "pnp_fallback": True,
        "pnp_holdout_every": 5,
        "pnp_good_frac": 0.0011,
        "pnp_weak_frac": 0.0027,
        "pnp_keep_failed": True,
        "pnp_fov_x_deg": None,  # horizontal FOV of the plate; None = MoGe-3's estimate
        "pnp_max_corners": 6000,
        # FOV sweep x0.85..x1.15 scored on held-out points (skipped when
        # pnp_fov_x_deg pins the FOV); depth gate on the anchor points and
        # an extra margin around tracked objects (of the long edge).
        "pnp_fov_sweep": True,
        "pnp_depth_range_m": (0.2, 60.0),
        "pnp_mask_grow_frac": 0.005,
        # "auto": LK, then LK guided by the job's flow pass (MegaFlow / RAFT)
        # when LK's camera is not GOOD; "klt" / "flow" force one of them.
        "pnp_tracker": "auto",
        # COLMAP (SIFT, matching, mapping) runs in its own process: a native
        # crash in pycolmap then loses the camera only, not every pass of the
        # run, and the fallbacks still get their turn. 0 = no time limit.
        "colmap_subprocess": True,
        "colmap_timeout_s": 0,
        "keep_workdir": False,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._solve: CameraSolve | None = None
        self._hard_masks: dict[int, dict[str, Any]] = {}
        self._workdir: Path | None = None
        self._sparse_dir: Path | None = None
        self._masked_labels: list[str] = []
        self._lens_mm: float | None = None
        self._pnp_fov_source = ""

    # --- per-frame hooks unused (VIDEO_CLIP) ---
    def preprocess(self, frames: np.ndarray) -> Any:  # pragma: no cover
        raise NotImplementedError("camera_pycolmap is VIDEO_CLIP; use run_shot().")

    def infer(self, tensor: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("camera_pycolmap is VIDEO_CLIP; use run_shot().")

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:  # pragma: no cover
        raise NotImplementedError("camera_pycolmap is VIDEO_CLIP; use run_shot().")

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        hard = artifacts.get("sam3_hard_masks") or {}
        self._hard_masks = next(iter(hard.values())) if hard else {}
        # A flow pass in the job (MegaFlow / RAFT): PnP tracks on its vectors.
        self._fwd_flow = dict(artifacts.get("forward_flow") or {})
        self._bwd_flow = dict(artifacts.get("backward_flow") or {})

    # --- shot entry point ---
    def run_shot(
        self, reader: Any, frame_range: tuple[int, int]
    ) -> dict[int, dict[str, np.ndarray]]:
        try:
            return self._run_shot(reader, frame_range)
        except BaseException:
            # The workdir holds plate frames: gone on any failure, whoever
            # the caller is (emit_sidecars only runs after a success).
            self._drop_workdir()
            raise

    def _run_shot(
        self, reader: Any, frame_range: tuple[int, int]
    ) -> dict[int, dict[str, np.ndarray]]:
        try:
            import pycolmap  # noqa: F401
        except ImportError as e:  # pragma: no cover — optional extra
            raise ImportError(
                "pycolmap is required for the camera track. Install with: "
                "uv sync --extra all   (or pip install 'live-action-aov[camera]')"
            ) from e
        model = str(self.params["camera_model"]).upper()
        if model not in _SUPPORTED_MODELS:
            raise ValueError(f"camera_model must be one of {_SUPPORTED_MODELS}, got {model!r}")

        self._workdir = make_work_dir("laaov_colmap_")
        image_dir = self._workdir / "images"
        image_dir.mkdir(parents=True)
        frame_list, plate_wh, solve_wh = self._export_frames(reader, frame_range, image_dir)
        if len(frame_list) < 3:
            raise RuntimeError(f"Camera track needs at least 3 frames; got {len(frame_list)}.")
        mask_dir = self._export_masks(frame_list, solve_wh)
        min_frac = float(self.params.get("min_registered_fraction", 0.5))
        failure: RuntimeError | None = None
        try:
            rec = self._solve_colmap(self._workdir, image_dir, mask_dir, frame_list)
            self._solve = build_solve_from_reconstruction(
                rec,
                frame_list=frame_list,
                plate_wh=plate_wh,
                params=self.params,
                masked_labels=self._masked_labels,
                verdict=self._referee_sfm(rec, frame_list, solve_wh, image_dir, mask_dir),
            )
        except RuntimeError as e:
            failure, self._solve = e, None
        short = (
            self._solve is not None
            and self._solve.quality.registered_frame_count
            < min_frac * self._solve.quality.total_frame_count
        )
        if (self._solve is None or short) and self.params.get("nodal_fallback", True):
            nodal = self._solve_nodal(self._workdir / "database.db", frame_list, plate_wh, solve_wh)
            if nodal is not None:
                self._solve, self._sparse_dir = nodal, None
                return {}
        if (self._solve is None or short) and self.params.get("pnp_fallback", True):
            # COLMAP's focal from a partial solve beats MoGe-3's single-frame
            # FOV (often 20-35 % wide on real plates).
            self._pnp_fov_hint = _fov_x_deg(self._solve) if self._solve is not None else None
            pnp_solve = self._solve_pnp(reader, frame_list, plate_wh, solve_wh, image_dir, mask_dir)
            if pnp_solve is not None:
                self._solve, self._sparse_dir = pnp_solve, None
                return {}
        if self._solve is None:
            raise failure or RuntimeError("Camera track: no reconstruction.")
        q = self._solve.quality
        _log.info(
            "Camera solve: %d/%d frames registered, mean reprojection %.3f px (plate res), "
            "focal %.1f px; masked: %s",
            q.registered_frame_count,
            q.total_frame_count,
            q.aggregate_reprojection_error_px,
            self._solve.intrinsics.fx_px,
            ", ".join(self._masked_labels) or "none",
        )
        for w in q.warnings:
            _log.warning("Camera solve: %s", w)
        if q.registered_frame_count < min_frac * q.total_frame_count:
            # A solve covering a few frames is not a camera for the shot
            # (no parallax over most of it: a pan or a locked-off camera).
            self._solve = None
            raise RuntimeError(
                f"Camera track: only {q.registered_frame_count}/{q.total_frame_count} frames "
                "could be solved - most of the shot has no parallax (pure pan, locked-off "
                "or very slow camera). No camera written."
            )
        return {}

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if self._solve is None or not self._solve.extrinsics:
            return {}
        # Shot-level artifact, stored under the first solved frame.
        return {"camera_solve": {self._solve.extrinsics[0].frame: self._solve}}

    def emit_sidecars(self, shot: Any) -> dict[str, Path]:
        if self._solve is None:
            return {}
        self._solve.shot = ShotRef(
            name=shot.name,
            plate_source=str(shot.folder),
            frame_range=tuple(shot.frame_range),
            resolution=tuple(shot.resolution),
            pixel_aspect=float(shot.pixel_aspect),
        )
        out_dir = sub_dir(shot, CAMERA_DIR)
        stem = run_stem(shot)
        json_path = out_dir / file_name(shot, "camera", "json")
        nk_path = out_dir / file_name(shot, "camera", "nk")
        ply_path = out_dir / file_name(shot, "points", "ply")
        obj_path = out_dir / file_name(shot, "points", "obj")
        json_path.write_text(self._solve.model_dump_json(indent=2), encoding="utf-8")
        nk_path.write_text(render_nuke_camera_script(self._solve), encoding="utf-8")
        write_points_ply(self._solve, ply_path)
        write_points_obj(self._solve, obj_path)
        # The same camera for Houdini, Maya, Blender, Unreal: plain-text USD.
        # The run rewrites it at the end with the cloud made for looking at.
        usd_path = write_usd(out_dir / file_name(shot, "camera", "usda"), self._solve)
        paths: dict[str, Path] = {
            "camera": json_path,
            "camera_nk": nk_path,
            "camera_usd": usd_path,
            "camera_points": ply_path,
            "camera_points_obj": obj_path,
        }
        if self._sparse_dir is not None and self._sparse_dir.exists():
            model_dir = out_dir / f"{stem}_colmap"
            shutil.rmtree(model_dir, ignore_errors=True)
            shutil.copytree(self._sparse_dir, model_dir)
            paths["camera_colmap"] = model_dir
        self._drop_workdir()
        return paths

    def unload(self) -> None:
        # Safety net when emit_sidecars never ran (failed shot, COLMAP crash).
        self._drop_workdir()

    def _drop_workdir(self) -> None:
        if self._workdir is None:
            return
        if self.params.get("keep_workdir"):
            # It holds frames exported from the plate: outside the project
            # tree, so say where it is.
            _log.warning("COLMAP workdir kept (contains plate frames): %s", self._workdir)
        else:
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None

    # --- internals ---
    def _export_frames(
        self, reader: Any, frame_range: tuple[int, int], image_dir: Path
    ) -> tuple[list[tuple[int, str]], tuple[int, int], tuple[int, int]]:
        import cv2

        max_size = int(self.params["max_image_size"])
        clahe = (
            cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            if self.params.get("contrast_normalize", True)
            else None
        )
        out: list[tuple[int, str]] = []
        plate_wh = solve_wh = (0, 0)
        read = reader.read_frame
        full_read = getattr(reader, "read_frame_full_res", None)
        full_wh = getattr(reader, "full_res_resolution", None)
        if callable(full_read) and callable(full_wh):
            proxy_long, plate_long = max(reader.resolution()), max(full_wh())
            if proxy_long < min(max_size, plate_long):
                # The proxy would cap `max_image_size`: read the plate itself.
                read = full_read
        for f in range(frame_range[0], frame_range[1] + 1):
            arr, attrs = read(f)
            if f == frame_range[0]:
                self._lens_mm = lens_focal_mm(attrs)
            arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                grey = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
            else:
                grey = arr.reshape(arr.shape[0], arr.shape[1])
            g8 = (np.clip(grey, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            h, w = g8.shape
            plate_wh = (w, h)
            s = min(1.0, max_size / max(h, w))
            sw, sh = max(1, round(w * s)), max(1, round(h * s))
            if (sw, sh) != (w, h):
                g8 = cv2.resize(g8, (sw, sh), interpolation=cv2.INTER_AREA)
            solve_wh = (sw, sh)
            if clahe is not None:
                g8 = clahe.apply(g8)
            name = f"frame_{f:06d}.png"
            cv2.imwrite(str(image_dir / name), g8)
            out.append((f, name))
        # Behind a proxy the frames are smaller than the plate: px reported
        # "at plate res" (errors, intrinsics) must use the plate's own size.
        full = getattr(reader, "full_res_resolution", None)
        if callable(full):
            plate_wh = tuple(full())
        return out, plate_wh, solve_wh

    def _export_masks(
        self, frame_list: list[tuple[int, str]], solve_wh: tuple[int, int]
    ) -> Path | None:
        """Union of SAM 3 object masks per frame -> COLMAP masks (0 = ignore)."""
        self._masked_labels = []
        if not self.params.get("mask_objects", True) or not self._hard_masks:
            return None
        import cv2

        wanted = {str(x).lower() for x in (self.params.get("mask_labels") or [])}
        tracks = [
            t
            for t in self._hard_masks.values()
            if not wanted or str(t.get("label", "")).lower() in wanted
        ]
        if not tracks:
            return None
        self._masked_labels = sorted({str(t.get("label", "")) for t in tracks})
        sw, sh = solve_wh
        r = max(1, round(float(self.params.get("mask_dilate_frac", 0.01)) * max(sw, sh)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        index = [{fr: i for i, fr in enumerate(t.get("frames") or [])} for t in tracks]
        assert self._workdir is not None
        mask_dir = self._workdir / "masks"
        mask_dir.mkdir(exist_ok=True)
        for f, name in frame_list:
            union = np.zeros((sh, sw), np.uint8)
            for t, idx in zip(tracks, index, strict=True):
                i = idx.get(f)
                if i is None:
                    continue
                m = np.asarray(t["stack"][i]) > 0
                if m.shape != (sh, sw):
                    m = (
                        cv2.resize(m.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
                        > 0
                    )
                union |= m.astype(np.uint8)
            excluded = cv2.dilate(union, kernel) if union.any() else union
            keep = np.where(excluded > 0, 0, 255).astype(np.uint8)
            # COLMAP looks up `<mask_path>/<image name>.png`.
            cv2.imwrite(str(mask_dir / f"{name}.png"), keep)
        return mask_dir

    def _solve_colmap(
        self,
        workdir: Path,
        image_dir: Path,
        mask_dir: Path | None,
        frame_list: list[tuple[int, str]],
    ) -> Any:
        """COLMAP reconstruction (pycolmap `Reconstruction`); sets
        `_sparse_dir`. In a child process unless `colmap_subprocess` is off."""
        names = [n for _, n in frame_list]
        if not self.params.get("colmap_subprocess", True):
            rec, self._sparse_dir = run_colmap(workdir, image_dir, mask_dir, names, self.params)
            return rec
        model_dir = run_colmap_subprocess(
            workdir,
            image_dir,
            mask_dir,
            names,
            self.params,
            timeout_s=float(self.params.get("colmap_timeout_s", 0) or 0),
        )
        pycolmap: Any = importlib.import_module("pycolmap")
        self._sparse_dir = model_dir
        return pycolmap.Reconstruction(str(model_dir))

    def _solve_nodal(
        self,
        db_path: Path,
        frame_list: list[tuple[int, str]],
        plate_wh: tuple[int, int],
        solve_wh: tuple[int, int],
    ) -> CameraSolve | None:
        """Rotation-only solve from COLMAP's verified matches, or None when
        it does not fit (the camera really translates, or too few matches)."""
        from live_action_aov.passes.camera import nodal

        if not db_path.is_file():  # COLMAP crashed before writing its database
            return None
        try:
            pairs = read_verified_pairs(db_path, frame_list)
        except RuntimeError as e:  # a database left half-written by a crash
            _log.info("Camera solve: nodal fallback cannot read COLMAP's matches (%s).", e)
            return None
        if not pairs:
            return None
        w, h = solve_wh
        c = (w / 2.0, h / 2.0)
        long_edge = float(max(w, h))
        hap = float(self.params.get("haperture_mm", 24.892))
        prior = self._lens_mm / hap * w if self._lens_mm else 0.8 * long_edge
        frames = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
        R0 = nodal.chain_rotations(pairs, prior, c, frames)
        angle = max(float(np.linalg.norm(nodal.matrix_to_rotvec(R))) for R in R0.values())
        measured = angle * prior >= float(self.params.get("nodal_min_motion", 0.03)) * long_edge
        f0 = prior
        if measured:
            f0 = nodal.estimate_focal(pairs, solve_wh)
            R0 = nodal.chain_rotations(pairs, f0, c, frames)
        sol = nodal.refine(pairs, f0, R0, solve_wh, fix_focal=not measured)
        errs = list(sol.frame_error_px.values())
        med = float(np.median(errs)) if errs else 1e9
        need = float(self.params.get("min_registered_fraction", 0.5)) * len(frame_list)
        if med > float(self.params.get("nodal_max_error_px", 1.5)) or len(sol.rotations) < need:
            _log.info(
                "Camera solve: rotation-only fallback does not fit (median %.2f px, %d/%d "
                "frames) - the camera translates or the plate has too little detail.",
                med,
                len(sol.rotations),
                len(frame_list),
            )
            return None
        if measured:
            focal_note = f"focal measured from the pan ({sol.focal_px * plate_wh[0] / w:.0f} px)"
        elif self._lens_mm:
            focal_note = (
                f"the camera barely moves, so the focal comes from the lens metadata "
                f"({self._lens_mm:g} mm on a {hap:g} mm film back - set the real film back in Nuke)"
            )
        else:
            focal_note = (
                "the camera barely moves and the plate has no lens metadata: the focal is a "
                "GUESS (0.8 x long edge) - set it from the camera report"
            )
        solve = build_solve_from_nodal(
            sol,
            frame_list=frame_list,
            plate_wh=plate_wh,
            solve_wh=solve_wh,
            params=self.params,
            masked_labels=self._masked_labels,
            pairs=pairs,
            lens_mm=None if measured else self._lens_mm,
            focal_source="nodal_pan"
            if measured
            else ("lens_metadata" if self._lens_mm else "guess"),
        )
        solve.quality.warnings.insert(
            0, f"No parallax: rotation-only (nodal) camera, translation fixed at 0; {focal_note}."
        )
        q = solve.quality
        _log.info(
            "Camera solve (nodal, rotation only): %d/%d frames, %.3f px (plate res), "
            "focal %.1f px; masked: %s",
            q.registered_frame_count,
            q.total_frame_count,
            q.aggregate_reprojection_error_px,
            solve.intrinsics.fx_px,
            ", ".join(self._masked_labels) or "none",
        )
        for warn in q.warnings:
            _log.warning("Camera solve: %s", warn)
        return solve

    def _solve_pnp(
        self,
        reader: Any,
        frame_list: list[tuple[int, str]],
        plate_wh: tuple[int, int],
        solve_wh: tuple[int, int],
        image_dir: Path,
        mask_dir: Path | None,
    ) -> CameraSolve | None:
        """Depth-anchored PnP (passes/camera/pnp.py), or None when it does
        not fit or MoGe-3 is not installed."""
        import cv2

        from live_action_aov.passes.camera import pnp

        sw, sh = solve_wh
        anchor = len(frame_list) // 2
        try:
            img = np.asarray(reader.read_frame(frame_list[anchor][0])[0], dtype=np.float32)[..., :3]
            depth, fov = self._anchor_depth(img)
        except (ImportError, RuntimeError) as e:
            _log.info("Camera solve: PnP fallback unavailable (%s).", e)
            return None
        depth = cv2.resize(depth, (sw, sh), interpolation=cv2.INTER_NEAREST)
        f = pnp.focal_from_fov(sw, fov)
        K = np.array([[f, 0.0, sw / 2.0], [0.0, f, sh / 2.0], [0.0, 0.0, 1.0]])
        gray, static = self._load_solve_frames(frame_list, solve_wh, image_dir, mask_dir)
        usable = ~pnp.near_depth_edges(depth, radius=11)  # = the LK window (21 px)
        # Sky / far background: the depth error there is huge.
        z_lo, z_hi = self.params.get("pnp_depth_range_m", (0.2, 60.0))
        usable &= (depth > float(z_lo)) & (depth < float(z_hi))
        anchor_static = static[anchor]
        if anchor_static is not None:
            usable &= anchor_static
        uv = pnp.anchor_corners(
            gray[anchor], usable, int(self.params.get("pnp_max_corners", 6000)), max(4, sw // 400)
        )
        if len(uv) < 50:
            _log.info("Camera solve: PnP fallback found only %d static corners.", len(uv))
            return None
        rgb = cv2.resize(np.clip(img, 0.0, 1.0), (sw, sh), interpolation=cv2.INTER_AREA)
        xi = np.clip(np.round(uv[:, 0]).astype(int), 0, sw - 1)
        yi = np.clip(np.round(uv[:, 1]).astype(int), 0, sh - 1)
        sol = pnp.PnPSolve(
            K=K,
            anchor=anchor,
            points=pnp.back_project(uv, depth, K),
            colors=(rgb[yi, xi] * 255.0 + 0.5).astype(np.uint8),
        )
        held = pnp.holdout_mask(len(uv), int(self.params.get("pnp_holdout_every", 5)))
        min_frac = float(self.params.get("min_registered_fraction", 0.5))
        good_frac = float(self.params.get("pnp_good_frac", 0.0011))
        weak_frac = float(self.params.get("pnp_weak_frac", 0.0027))
        rank = {"GOOD": 2, "WEAK": 1, "FAILED": 0}

        def attempt(
            Kt: np.ndarray, tracks: np.ndarray, sub: list[int] | None = None
        ) -> tuple[pnp.PnPSolve, pnp.PnPVerdict | None]:
            tr, anc, n = (tracks, anchor, len(frame_list))
            if sub is not None:
                tr, anc, n = tracks[sub], sub.index(anchor), len(sub)
            trial = pnp.PnPSolve(
                K=Kt, anchor=anc, points=pnp.back_project(uv, depth, Kt), colors=sol.colors
            )
            pnp.solve_poses(trial, tr, held=held)
            if len(trial.poses) < min_frac * n:
                return trial, None
            return trial, pnp.assess(trial, sw, n, good_frac=good_frac, weak_frac=weak_frac)

        def score(v: pnp.PnPVerdict | None) -> tuple[float, ...]:
            return (-1.0,) if v is None else (rank[v.verdict], len(v.reliable), -v.heldout_px)

        best: tuple[tuple[float, ...], pnp.PnPSolve, pnp.PnPVerdict | None, np.ndarray] | None
        best = None
        for guide in self._pnp_guides(frame_list, sw):
            tracks = self._track_pnp(gray, anchor, uv, static, guide)
            trial, v = attempt(K, tracks)
            if best is None or score(v) > best[0]:
                best = (score(v), trial, v, tracks)
            if v is not None and v.verdict == "GOOD":
                break
        assert best is not None
        if (
            best[2] is not None
            and self.params.get("pnp_fov_x_deg") is None
            and self.params.get("pnp_fov_sweep", True)
        ):
            fac = self._fov_sweep(attempt, best[3], K, fov, sw, weak_frac)
            if fac != 1.0:
                Kw = K.copy()
                Kw[0, 0] = Kw[1, 1] = pnp.focal_from_fov(sw, fov * fac)
                trial, v = attempt(Kw, best[3])
                if score(v) >= best[0]:
                    best = (score(v), trial, v, best[3])
                    self._pnp_fov_source = (
                        f"FOV {fov * fac:.1f} deg = x{fac:.2f} of {self._pnp_fov_source} "
                        "(held-out sweep)"
                    )
        assert best is not None
        sol, verdict = best[1], best[2]
        if verdict is None:
            _log.warning(
                "Camera solve: depth-anchored PnP FAILED - point tracks lost, only %d/%d frames "
                "solved (fast motion or blur).",
                len(sol.poses),
                len(frame_list),
            )
            return None
        if verdict.verdict == "FAILED" and not self.params.get("pnp_keep_failed", True):
            _log.info(
                "Camera solve: depth-anchored PnP FAILED (held-out %.2f px at solve res).",
                verdict.heldout_px,
            )
            return None
        if verdict.verdict != "FAILED":
            # Unreliable frames would make the camera jump: leave them out
            # (Nuke interpolates over the gap).
            keep = set(verdict.reliable)
            sol.poses = {k: v for k, v in sol.poses.items() if k in keep}
        solve = build_solve_from_pnp(
            sol,
            frame_list=frame_list,
            plate_wh=plate_wh,
            solve_wh=solve_wh,
            params=self.params,
            masked_labels=self._masked_labels,
            fov_note=self._pnp_fov_source,
            verdict=verdict,
        )
        q = solve.quality
        _log.info(
            "Camera solve (depth-anchored PnP): %d/%d frames, %.3f px (plate res), focal %.1f px "
            "(%s); masked: %s",
            q.registered_frame_count,
            q.total_frame_count,
            q.aggregate_reprojection_error_px,
            solve.intrinsics.fx_px,
            self._pnp_fov_source,
            ", ".join(self._masked_labels) or "none",
        )
        for warn in q.warnings:
            _log.warning("Camera solve: %s", warn)
        return solve

    def _load_solve_frames(
        self,
        frame_list: list[tuple[int, str]],
        solve_wh: tuple[int, int],
        image_dir: Path,
        mask_dir: Path | None,
    ) -> tuple[list[np.ndarray], list[np.ndarray | None]]:
        """The exported solve-resolution frames (uint8 grey) and, per frame,
        the static mask (True = not a tracked object) or None. Matte edges
        (motion blur, hair) poison a track: the mask keeps a margin around
        the objects on top of the COLMAP mask dilation."""
        import cv2

        sw, sh = solve_wh
        gray = [cv2.imread(str(image_dir / name), cv2.IMREAD_GRAYSCALE) for _, name in frame_list]
        static: list[np.ndarray | None] = [
            cv2.imread(str(mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 0
            if mask_dir is not None and (mask_dir / f"{name}.png").is_file()
            else None
            for _, name in frame_list
        ]
        grow = round(float(self.params.get("pnp_mask_grow_frac", 0.005)) * max(sw, sh))
        if grow > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
            static = [
                cv2.erode(m.astype(np.uint8), kernel) > 0 if m is not None else None for m in static
            ]
        return gray, static

    def _referee_sfm(
        self,
        reconstruction: Any,
        frame_list: list[tuple[int, str]],
        solve_wh: tuple[int, int],
        image_dir: Path,
        mask_dir: Path | None,
    ) -> Any:
        """Held-out verdict (`pnp.PnPVerdict`) of COLMAP's camera, so an SfM
        solve is judged like the depth-anchored one: KLT corners of the
        middle registered frame tracked through the shot (tracked objects
        masked out), each one triangulated with COLMAP's poses from every
        frame but the one under test and reprojected into it
        (`pnp.heldout_from_poses`). Trackers as the PnP path tries them
        (`pnp_tracker`); the best verdict wins, GOOD stops the search."""
        from live_action_aov.passes.camera import pnp

        sw, _sh = solve_wh
        idx = {name: k for k, (_f, name) in enumerate(frame_list)}
        poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        cam_counts: dict[int, int] = {}
        for im in reconstruction.images.values():
            k = idx.get(im.name)
            if k is None or not _is_registered(im):
                continue
            pose = _pose(im)
            poses[k] = (
                np.asarray(pose.rotation.matrix(), dtype=np.float64),
                np.asarray(pose.translation, dtype=np.float64).reshape(3),
            )
            cam_counts[im.camera_id] = cam_counts.get(im.camera_id, 0) + 1
        n = len(frame_list)
        good_frac = float(self.params.get("pnp_good_frac", 0.0011))
        weak_frac = float(self.params.get("pnp_weak_frac", 0.0027))
        if len(poses) < 2:
            # Nothing to referee: `assess` on an empty solve says FAILED.
            empty = pnp.PnPSolve(
                K=np.eye(3), anchor=0, points=np.zeros((0, 3)), colors=np.zeros((0, 3), np.uint8)
            )
            return pnp.assess(empty, sw, n, good_frac=good_frac, weak_frac=weak_frac)
        camera = reconstruction.cameras[max(cam_counts, key=lambda c: cam_counts[c])]
        intr = _intrinsics(camera, (0, 0), self.params)  # solve resolution; k1 not applied
        K = np.array(
            [[intr.fx_px, 0.0, intr.cx_px], [0.0, intr.fy_px, intr.cy_px], [0.0, 0.0, 1.0]]
        )
        anchor = min(poses, key=lambda k: (abs(k - n // 2), k))
        gray, static = self._load_solve_frames(frame_list, solve_wh, image_dir, mask_dir)
        anchor_static = static[anchor]
        usable = anchor_static if anchor_static is not None else np.ones(gray[anchor].shape, bool)
        uv = pnp.anchor_corners(
            gray[anchor], usable, int(self.params.get("pnp_max_corners", 6000)), max(4, sw // 400)
        )
        rank = {"GOOD": 2, "WEAK": 1, "FAILED": 0}
        best: pnp.PnPVerdict | None = None
        for guide in self._pnp_guides(frame_list, sw):
            tracks = self._track_pnp(gray, anchor, uv, static, guide)
            v = pnp.assess_poses(
                K, poses, tracks, anchor, sw, n, good_frac=good_frac, weak_frac=weak_frac
            )
            if best is None or (rank[v.verdict], len(v.reliable), -v.heldout_px) > (
                rank[best.verdict],
                len(best.reliable),
                -best.heldout_px,
            ):
                best = v
            if v.verdict == "GOOD":
                break
        assert best is not None
        _log.info(
            "Camera solve: KLT referee on the SfM camera - %s, held-out %.2f px (solve res), "
            "%d/%d reliable frames, %d corners",
            best.verdict,
            best.heldout_px,
            len(best.reliable),
            n,
            len(uv),
        )
        return best

    def _fov_sweep(
        self,
        attempt: Any,
        tracks: np.ndarray,
        K: np.ndarray,
        fov: float,
        sw: int,
        weak_frac: float,
    ) -> float:
        """FOV factor that fits the tracks better than the estimate, or 1.0.

        MoGe-3's single-frame FOV can be ~20 % off on real plates. Each factor
        is solved on every 3rd frame; another one wins only if it cuts the
        held-out error by >= 15 %, is not at the edge of the sweep, solves at
        least as many frames and is itself within the WEAK limit (on a
        failed solve the sweep is noise)."""
        from live_action_aov.passes.camera import pnp

        T = tracks.shape[0]
        anchor = T // 2
        sub = sorted(set(range(anchor % 3, T, 3)) | {anchor})
        factors = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15)
        res: dict[float, tuple[float, int]] = {}
        for fac in factors:
            Kf = K.copy()
            Kf[0, 0] = Kf[1, 1] = pnp.focal_from_fov(sw, fov * fac)
            trial, v = attempt(Kf, tracks, sub)
            if v is not None:
                res[fac] = (v.heldout_px, len(trial.poses))
        if 1.0 not in res:
            return 1.0
        win = min(res, key=lambda k: res[k][0])
        held, n = res[win]
        base_held, base_n = res[1.0]
        ok = (
            win not in (factors[0], factors[-1], 1.0)
            and held <= 0.85 * base_held
            and n >= base_n
            and held <= weak_frac * sw
        )
        _log.info(
            "Camera solve: PnP FOV sweep (every 3rd frame): %s -> %s",
            ", ".join(f"x{k:.2f} {h:.2f}px" for k, (h, _n) in sorted(res.items())),
            f"x{win:.2f}" if ok else "kept the estimate",
        )
        return win if ok else 1.0

    def _pnp_guides(self, frame_list: list[tuple[int, str]], sw: int) -> list[Any]:
        """Trackers to try, in order: None = plain LK, else a flow guide.

        `pnp_tracker: auto` tries LK first and, when the job has a flow pass
        (MegaFlow / RAFT) and LK's camera is not GOOD, LK guided by the flow
        vectors: it follows fast handheld motion that LK loses, but on slow
        shots its extra points are noisier, so it never replaces a GOOD LK."""
        from live_action_aov.passes.camera import pnp

        frames = [f for f, _ in frame_list]
        mode = str(self.params.get("pnp_tracker", "auto"))
        fwd = getattr(self, "_fwd_flow", {})
        bwd = getattr(self, "_bwd_flow", {})
        if mode == "klt" or not all(f in fwd and f in bwd for f in frames):
            if mode == "flow":
                _log.warning("Camera solve: pnp_tracker 'flow' but the job has no flow pass.")
            return [None]
        scale = sw / max(int(fwd[frames[0]].shape[-1]), 1)
        guide = pnp.flow_guide(fwd, bwd, frames, scale=scale)
        return [guide] if mode == "flow" else [None, guide]

    def _track_pnp(
        self,
        gray: list[np.ndarray],
        anchor: int,
        uv: np.ndarray,
        static: list[np.ndarray | None],
        guide: Any,
    ) -> np.ndarray:
        from live_action_aov.passes.camera import pnp

        tracks = pnp.track_klt(gray, anchor, uv, static, guide=guide)
        alive = np.isfinite(tracks[..., 0]).sum(axis=1)
        _log.info(
            "Camera solve: PnP tracks (%s), %d points; alive at the ends %d / %d",
            "LK guided by the flow pass" if guide is not None else "LK",
            len(uv),
            int(alive[0]),
            int(alive[-1]),
        )
        return tracks

    def _anchor_depth(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """MoGe-3 metric depth (H, W, 0 = invalid) of one display-referred
        frame and the horizontal FOV used. Unloaded right after."""
        from live_action_aov.passes.normals.moge3 import MoGe3Pass

        fov_param = self.params.get("pnp_fov_x_deg")
        hint = getattr(self, "_pnp_fov_hint", None)
        fov_in = fov_param if fov_param is not None else hint
        moge = MoGe3Pass({"emit_normals": False})
        try:
            r = moge.infer(
                moge.preprocess(img[None]), float(fov_in) if fov_in is not None else None
            )
        finally:
            moge._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover
                pass
        self._pnp_fov_source = f"FOV {float(r['fov_x']):.1f} deg " + (
            "from pnp_fov_x_deg"
            if fov_param is not None
            else "from COLMAP's partial solve"
            if hint is not None
            else "estimated by MoGe-3"
        )
        return r["depth"], float(r["fov_x"])


# ---------------------------------------------------------------------------
# COLMAP pipeline (in this process, or in a child via colmap_worker)
# ---------------------------------------------------------------------------

_COLMAP_PARAMS = (
    "camera_model",
    "sift_num_threads",
    "sift_max_features",
    "sift_peak_threshold",
    "matcher",
    "sequential_overlap",
    "mapper",
    "min_registered_fraction",
    "mapping_attempts",
)


def run_colmap(
    workdir: Path,
    image_dir: Path,
    mask_dir: Path | None,
    image_names: list[str],
    params: dict[str, Any],
) -> tuple[Any, Path]:
    """SIFT -> matching -> mapping; (largest reconstruction, its folder).
    Raises RuntimeError when nothing reconstructs."""
    pycolmap: Any = importlib.import_module("pycolmap")  # no usable type stubs

    db = workdir / "database.db"
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = str(params["camera_model"]).upper()
    if mask_dir is not None:
        reader_opts.mask_path = str(mask_dir)
    ext = pycolmap.FeatureExtractionOptions()
    ext.sift.max_num_features = int(params["sift_max_features"])
    ext.sift.peak_threshold = float(params["sift_peak_threshold"])
    threads = int(params.get("sift_num_threads", 0) or 0)
    w, h = _image_size(image_dir / image_names[0])
    if threads <= 0:
        threads = sift_threads_for_ram(w, h, _free_ram_gb(), os.cpu_count() or 1)
    ext.num_threads = threads
    _log.info(
        "Camera solve: extracting features on %d frames at %dx%d, %d threads",
        len(image_names),
        w,
        h,
        threads,
    )
    pycolmap.extract_features(
        database_path=db,
        image_path=image_dir,
        image_names=list(image_names),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_opts,
        extraction_options=ext,
    )
    _log.info("Camera solve: matching (%s)", params["matcher"])
    if params["matcher"] == "exhaustive":
        pycolmap.match_exhaustive(db)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = int(params["sequential_overlap"])
        pycolmap.match_sequential(db, pairing_options=pairing)
    sparse = workdir / "sparse"
    sparse.mkdir(exist_ok=True)
    _log.info("Camera solve: mapping")
    if params.get("mapper") == "global":
        recs = pycolmap.global_mapping(database_path=db, image_path=image_dir, output_path=sparse)
        return _pick_model(recs, sparse)
    need = float(params.get("min_registered_fraction", 0.5)) * len(image_names)
    attempts = max(1, int(params.get("mapping_attempts", 1)))
    best: tuple[Any, Path] | None = None
    for seed in range(attempts):
        out = sparse / f"seed{seed}"
        out.mkdir(exist_ok=True)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.random_seed = seed
        recs = pycolmap.incremental_mapping(
            database_path=db, image_path=image_dir, output_path=out, options=opts
        )
        if recs:
            rec, model_dir = _pick_model(recs, out)
            if best is None or rec.num_reg_images() > best[0].num_reg_images():
                best = (rec, model_dir)
            if rec.num_reg_images() >= need:
                break
        got = best[0].num_reg_images() if best else 0
        if seed + 1 < attempts:
            _log.info(
                "Camera solve: mapping seed %d covered %d/%d frames, retrying",
                seed,
                got,
                len(image_names),
            )
    if best is None:
        _no_reconstruction()
    return best


def run_colmap_subprocess(
    workdir: Path,
    image_dir: Path,
    mask_dir: Path | None,
    image_names: list[str],
    params: dict[str, Any],
    timeout_s: float = 0.0,
) -> Path:
    """`run_colmap` in a child Python (`colmap_worker`): a native crash
    (pycolmap can segfault in SIFT extraction on long shots) becomes a
    RuntimeError here instead of killing the whole run. Returns the model
    folder."""
    job = workdir / "colmap_job.json"
    result = workdir / "colmap_result.json"
    result.unlink(missing_ok=True)
    job.write_text(
        json.dumps(
            {
                "workdir": str(workdir),
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir) if mask_dir is not None else None,
                "image_names": list(image_names),
                "params": {k: params[k] for k in _COLMAP_PARAMS if k in params},
                "result": str(result),
            }
        ),
        encoding="utf-8",
    )
    cmd = [sys.executable, "-m", "live_action_aov.passes.camera.colmap_worker", str(job)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        creationflags=flags,
    )
    # Line by line, so a 30-minute SIFT run shows its progress live.
    killed = threading.Event()

    def _kill() -> None:
        killed.set()
        proc.kill()

    timer = threading.Timer(timeout_s, _kill) if timeout_s else None
    if timer is not None:
        timer.start()
    tail: list[str] = []
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            if line.startswith("Camera solve:"):
                _log.info("%s", line)
            elif line:
                tail = [*tail[-2:], line]
        code = proc.wait()
    finally:
        if timer is not None:
            timer.cancel()
    if killed.is_set() and not result.is_file():
        raise RuntimeError(f"Camera track: COLMAP did not finish in {timeout_s:.0f} s.")
    if code != 0 or not result.is_file():
        # Windows reports an access violation as 0xC0000005, POSIX as -signal.
        shown = f"signal {-code}" if code < 0 else f"0x{code:08X}" if code > 255 else str(code)
        raise RuntimeError(
            f"Camera track: COLMAP crashed (exit code {shown}); the other passes are not "
            f"affected. {' | '.join(tail)}".strip()
        )
    out = json.loads(result.read_text(encoding="utf-8"))
    if "error" in out:
        raise RuntimeError(out["error"])
    return Path(out["model_dir"])


#: SIFT extraction RAM per thread, GB per megapixel of the solve image
#: (measured: ~0.45 GB at 1920, ~1.3 at 2816, ~1.65 at 3840 px wide).
_SIFT_GB_PER_MPIX = 0.3
#: Without psutil: a cap that fits a 16-32 GB box at ~2K, not every core.
_SIFT_THREADS_UNKNOWN_RAM = 8


def _fov_x_deg(solve: CameraSolve) -> float | None:
    """Horizontal FOV of a solve's camera, None when it has no usable focal."""
    intr = solve.intrinsics
    if intr.fx_px <= 0 or intr.width_px <= 0:
        return None
    return float(np.degrees(2.0 * np.arctan(intr.width_px / (2.0 * intr.fx_px))))


def sift_threads_for_ram(w: int, h: int, free_gb: float | None, cpus: int) -> int:
    """SIFT threads that fit in `free_gb` (keeping 20 % headroom); at most
    `_SIFT_THREADS_UNKNOWN_RAM` when the free RAM cannot be read."""
    if free_gb is None:
        return max(1, min(cpus, _SIFT_THREADS_UNKNOWN_RAM))
    per_thread = 0.1 + _SIFT_GB_PER_MPIX * w * h / 1e6
    return max(1, min(cpus, int(0.8 * free_gb / per_thread)))


def _free_ram_gb() -> float | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is in the camera extra
        return None
    return float(psutil.virtual_memory().available) / 1e9


def _image_size(path: Path) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return (int(img.shape[1]), int(img.shape[0])) if img is not None else (0, 0)


def _pick_model(recs: dict[int, Any], out: Path) -> tuple[Any, Path]:
    """Largest of the mapper's models and its folder."""
    if not recs:
        _no_reconstruction()
    key, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
    model_dir = out / str(key)
    if not model_dir.exists():  # some pycolmap builds write only the best model
        model_dir.mkdir(parents=True, exist_ok=True)
        rec.write(str(model_dir))
    return rec, model_dir


def _no_reconstruction() -> NoReturn:
    raise RuntimeError(
        "Camera track: no reconstruction. Likely causes: no parallax "
        "(locked-off / pure pan), heavy motion blur, or a textureless plate."
    )


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without pycolmap)
# ---------------------------------------------------------------------------


def opencv_to_opengl_world_from_camera(
    R_cw: np.ndarray, t_cw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP `cam_from_world` (OpenCV camera) -> OpenGL `world_from_camera`.

    Both frames are flipped: the camera-local frame (OpenCV -> OpenGL) AND
    the world. COLMAP's world is the first camera's OpenCV frame (Y down,
    Z forward); rotating it 180 deg about X makes it Y-up with the camera
    looking down -Z, so the scene is upright in Nuke. Points get the same
    world flip (`world_point_to_gl`), so cameras and cloud stay consistent.

    Returns `(R_wc_gl, camera_world_position)`.
    """
    R_cw = np.asarray(R_cw, dtype=np.float64).reshape(3, 3)
    t_cw = np.asarray(t_cw, dtype=np.float64).reshape(3)
    F = _OPENCV_TO_OPENGL_FLIP
    R_wc_cv = R_cw.T
    return F @ R_wc_cv @ F, F @ (-R_wc_cv @ t_cw)


def rotation_matrix_to_euler_zxy_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose `R = Ry(b) @ Rx(a) @ Rz(c)` into `(a, b, c)` degrees —
    Nuke's `rot_order ZXY` (Rz applied first to local vectors)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    alpha = math.asin(max(-1.0, min(1.0, float(-R[1, 2]))))
    if abs(math.cos(alpha)) > 1e-6:
        beta = math.atan2(float(R[0, 2]), float(R[2, 2]))
        gamma = math.atan2(float(R[1, 0]), float(R[1, 1]))
    else:  # gimbal lock: pin gamma
        gamma = 0.0
        beta = math.atan2(-float(R[2, 0]), float(R[0, 0]))
    return math.degrees(alpha), math.degrees(beta), math.degrees(gamma)


def _pose(img: Any) -> Any:
    """`cam_from_world` is a property in pycolmap 3.x, a method in 4.x."""
    cfw = img.cam_from_world
    return cfw() if callable(cfw) else cfw


def _is_registered(img: Any) -> bool:
    has = getattr(img, "has_pose", None)
    if has is None:
        return True
    return bool(has() if callable(has) else has)


def _intrinsics(camera: Any, plate_wh: tuple[int, int], params: dict[str, Any]) -> Intrinsics:
    """Read a pycolmap camera and rescale it from solve to plate resolution."""
    model = str(getattr(camera, "model_name", None) or getattr(camera, "model", "")).upper()
    model = model.split(".")[-1]
    p = [float(x) for x in camera.params]
    dist: list[float] = []
    if model == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
    elif model == "SIMPLE_RADIAL":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
        dist = [p[3]]
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model {model}; use one of {_SUPPORTED_MODELS}."
        )
    sw, sh = int(camera.width), int(camera.height)
    pw, ph = plate_wh if plate_wh[0] > 0 else (sw, sh)
    sx, sy = pw / max(sw, 1), ph / max(sh, 1)
    fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
    hap = float(params.get("haperture_mm", 24.892))
    return Intrinsics(
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        width_px=pw,
        height_px=ph,
        haperture_mm_hint=hap,
        vaperture_mm_hint=hap * ph / max(pw, 1),
        focal_mm_hint=fx * hap / max(pw, 1),
        distortion_model="simple_radial" if dist else "pinhole",
        distortion_params=dist,
    )


def build_solve_from_reconstruction(
    reconstruction: Any,
    *,
    frame_list: list[tuple[int, str]],
    plate_wh: tuple[int, int] = (0, 0),
    params: dict[str, Any],
    masked_labels: list[str] | None = None,
    verdict: Any,
) -> CameraSolve:
    """pycolmap `Reconstruction` -> `CameraSolve` (OpenGL poses, Nuke Eulers,
    plate-resolution intrinsics, per-frame reprojection error in plate px).

    `verdict` (`pnp.PnPVerdict`, from `PyColmapCameraPass._referee_sfm`) is
    not optional: COLMAP's reprojection error is measured on the features
    it fitted, so every SfM camera carries the independent held-out verdict
    that leads its warnings and the Nuke label, like the PnP camera."""
    if verdict is None:
        raise TypeError("build_solve_from_reconstruction needs the referee's verdict.")
    name_to_frame = {name: frame for frame, name in frame_list}
    images = [
        im
        for im in reconstruction.images.values()
        if _is_registered(im) and im.name in name_to_frame
    ]
    cam_counts: dict[int, int] = {}
    for im in images:
        cam_counts[im.camera_id] = cam_counts.get(im.camera_id, 0) + 1
    if not cam_counts:
        raise RuntimeError("Camera track: reconstruction has no registered frames.")
    camera = reconstruction.cameras[max(cam_counts, key=lambda k: cam_counts[k])]
    intr = _intrinsics(camera, plate_wh, params)
    px_scale = intr.width_px / max(int(camera.width), 1)

    extrinsics: list[Extrinsic] = []
    per_frame_err: dict[int, float] = {}
    per_frame_count: dict[int, int] = {}
    for im in images:
        frame = name_to_frame[im.name]
        pose = _pose(im)
        R_cw = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        t_cw = np.asarray(pose.translation, dtype=np.float64)
        R_gl, t_gl = opencv_to_opengl_world_from_camera(R_cw, t_cw)
        extrinsics.append(
            Extrinsic(
                frame=frame,
                translation=(float(t_gl[0]), float(t_gl[1]), float(t_gl[2])),
                rotation_euler_zxy_deg=rotation_matrix_to_euler_zxy_deg(R_gl),
                rotation_matrix=tuple(tuple(float(v) for v in row) for row in R_gl),  # type: ignore[arg-type]
            )
        )
        errs = _image_reprojection_errors(reconstruction, im)
        if errs.size:
            per_frame_err[frame] = float(errs.mean() * px_scale)
            per_frame_count[frame] = int(errs.size)
    extrinsics.sort(key=lambda e: e.frame)

    points: list[ScenePoint] = []
    for pt in reconstruction.points3D.values():
        xyz = np.asarray(pt.xyz, dtype=np.float64)
        rgb = getattr(pt, "color", None)
        color = (255, 255, 255) if rgb is None else (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        track = getattr(pt, "track", None)
        points.append(
            ScenePoint(
                xyz=world_point_to_gl((float(xyz[0]), float(xyz[1]), float(xyz[2]))),
                color_rgb=color,
                reprojection_error_px=float(getattr(pt, "error", 0.0)) * px_scale,
                track_length=int(track.length())
                if track is not None and hasattr(track, "length")
                else 0,
            )
        )

    registered = {e.frame for e in extrinsics}
    total = len(frame_list)
    quality = _quality(per_frame_err, per_frame_count, len(registered), total, intr, extrinsics)
    _apply_verdict(quality, verdict, px_scale, total, metric=False)
    return CameraSolve(
        shot=ShotRef(),
        backend=BackendInfo(
            name="pycolmap",
            version=_pycolmap_version(),
            matcher=str(params.get("matcher", "sequential")),
            sequential_overlap=int(params.get("sequential_overlap", 10)),
            camera_model=str(params.get("camera_model", "SIMPLE_PINHOLE")),
            sift_max_features=int(params.get("sift_max_features", 8192)),
            max_image_size=int(params.get("max_image_size", 1920)),
            mapper=str(params.get("mapper", "incremental")),
            masked_labels=list(masked_labels or []),
        ),
        conventions=Conventions(),
        intrinsics=intr,
        extrinsics=extrinsics,
        unregistered_frames=sorted(f for f, _ in frame_list if f not in registered),
        points=points,
        quality=quality,
    )


def lens_focal_mm(attrs: dict[str, Any]) -> float | None:
    """Lens focal length (mm) from plate metadata (ARRI `.../FocalLength`)."""
    for k, v in (attrs or {}).items():
        if str(k).lower().endswith("focallength"):
            try:
                mm = float(v)
            except (TypeError, ValueError):
                continue
            if 1.0 < mm < 2000.0:
                return mm
    return None


def read_verified_pairs(
    db_path: Path, frame_list: list[tuple[int, str]], min_inliers: int = 30
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """(i, j, xi, xj) per verified COLMAP pair: indices into `frame_list`,
    inlier keypoint pixels at solve resolution."""
    pycolmap: Any = importlib.import_module("pycolmap")
    name_to_idx = {name: k for k, (_f, name) in enumerate(frame_list)}
    db = pycolmap.Database.open(str(db_path))
    try:
        ids = {im.image_id: name_to_idx.get(im.name) for im in db.read_all_images()}
        kp: dict[int, np.ndarray] = {}
        pair_ids, geoms = db.read_two_view_geometries()
        out = []
        for pid, g in zip(pair_ids, geoms, strict=False):
            a, b = int(pid) // 2147483647, int(pid) % 2147483647  # COLMAP pair id
            m = np.asarray(g.inlier_matches)
            i, j = ids.get(a), ids.get(b)
            if len(m) < min_inliers or i is None or j is None:
                continue
            for iid in (a, b):
                if iid not in kp:
                    kp[iid] = np.asarray(db.read_keypoints(iid))[:, :2].astype(np.float64)
            xi, xj = kp[a][m[:, 0]], kp[b][m[:, 1]]
            out.append((i, j, xi, xj) if i < j else (j, i, xj, xi))
        return out
    finally:
        db.close()


def build_solve_from_nodal(
    sol: Any,
    *,
    frame_list: list[tuple[int, str]],
    plate_wh: tuple[int, int],
    solve_wh: tuple[int, int],
    params: dict[str, Any],
    masked_labels: list[str] | None = None,
    pairs: list[tuple[int, int, np.ndarray, np.ndarray]] | None = None,
    lens_mm: float | None = None,
    focal_source: str = "nodal_pan",
    point_distance: float = 100.0,
    max_points: int = 4000,
) -> CameraSolve:
    """`nodal.NodalSolve` -> `CameraSolve`: rotation-only poses (camera at
    the origin), plate-resolution intrinsics, and the matched features as
    points on a sphere `point_distance` away (to check the pan in Nuke)."""
    sw, sh = solve_wh
    pw, ph = plate_wh if plate_wh[0] > 0 else solve_wh
    scale = pw / max(sw, 1)
    f = float(sol.focal_px) * scale
    hap = float(params.get("haperture_mm", 24.892))
    intr = Intrinsics(
        fx_px=f,
        fy_px=f,
        cx_px=pw / 2.0,
        cy_px=ph / 2.0,
        width_px=pw,
        height_px=ph,
        haperture_mm_hint=hap,
        vaperture_mm_hint=hap * ph / max(pw, 1),
        focal_mm_hint=float(lens_mm) if lens_mm else f * hap / max(pw, 1),
        distortion_model="pinhole",
        distortion_params=[],
        source=focal_source,
    )
    extrinsics: list[Extrinsic] = []
    per_err: dict[int, float] = {}
    per_count: dict[int, int] = {}
    for k, R_cw in sol.rotations.items():
        frame = frame_list[k][0]
        R_gl, t_gl = opencv_to_opengl_world_from_camera(R_cw, np.zeros(3))
        extrinsics.append(
            Extrinsic(
                frame=frame,
                translation=(float(t_gl[0]), float(t_gl[1]), float(t_gl[2])),
                rotation_euler_zxy_deg=rotation_matrix_to_euler_zxy_deg(R_gl),
                rotation_matrix=tuple(tuple(float(v) for v in row) for row in R_gl),  # type: ignore[arg-type]
            )
        )
        if k in sol.frame_error_px:
            per_err[frame] = float(sol.frame_error_px[k]) * scale
            per_count[frame] = int(sol.frame_match_count.get(k, 0))
    extrinsics.sort(key=lambda e: e.frame)
    points: list[ScenePoint] = []
    if pairs:
        from live_action_aov.passes.camera.nodal import bearings

        rng = np.random.default_rng(0)
        per_pair = max(1, max_points // len(pairs))
        for i, _j, xi, _xj in pairs:
            if i not in sol.rotations:
                continue
            sel = rng.choice(len(xi), min(per_pair, len(xi)), replace=False)
            dirs = bearings(xi[sel], float(sol.focal_px), (sw / 2.0, sh / 2.0)) @ sol.rotations[i]
            for d in dirs * point_distance:
                points.append(
                    ScenePoint(
                        xyz=world_point_to_gl((float(d[0]), float(d[1]), float(d[2]))),
                        color_rgb=(255, 255, 255),
                        reprojection_error_px=0.0,
                        track_length=2,
                    )
                )
    registered = {e.frame for e in extrinsics}
    quality = _quality(per_err, per_count, len(registered), len(frame_list), intr, extrinsics)
    return CameraSolve(
        shot=ShotRef(),
        backend=BackendInfo(
            name="pycolmap",
            version=_pycolmap_version(),
            matcher=str(params.get("matcher", "sequential")),
            sequential_overlap=int(params.get("sequential_overlap", 10)),
            camera_model="SIMPLE_PINHOLE",
            sift_max_features=int(params.get("sift_max_features", 8192)),
            max_image_size=int(params.get("max_image_size", 1920)),
            mapper="nodal",
            masked_labels=list(masked_labels or []),
        ),
        conventions=Conventions(),
        intrinsics=intr,
        extrinsics=extrinsics,
        unregistered_frames=sorted(fr for fr, _ in frame_list if fr not in registered),
        points=points,
        quality=quality,
    )


def build_solve_from_pnp(
    sol: Any,
    *,
    frame_list: list[tuple[int, str]],
    plate_wh: tuple[int, int],
    solve_wh: tuple[int, int],
    params: dict[str, Any],
    masked_labels: list[str] | None = None,
    fov_note: str = "",
    max_points: int = 30000,
    verdict: Any = None,
) -> CameraSolve:
    """`pnp.PnPSolve` -> `CameraSolve`: world = the anchor camera (OpenCV),
    flipped to Y-up like every other solve; metres from MoGe-3. `verdict`
    (`pnp.PnPVerdict`) leads the warnings, hence the Nuke camera label."""
    sw, _sh = solve_wh
    pw, ph = plate_wh if plate_wh[0] > 0 else solve_wh
    scale = pw / max(sw, 1)
    f = float(sol.K[0, 0]) * scale
    hap = float(params.get("haperture_mm", 24.892))
    intr = Intrinsics(
        fx_px=f,
        fy_px=f,
        cx_px=pw / 2.0,
        cy_px=ph / 2.0,
        width_px=pw,
        height_px=ph,
        haperture_mm_hint=hap,
        vaperture_mm_hint=hap * ph / max(pw, 1),
        focal_mm_hint=f * hap / max(pw, 1),
        distortion_model="pinhole",
        distortion_params=[],
        source="moge3_depth_pnp",
    )
    extrinsics: list[Extrinsic] = []
    per_err: dict[int, float] = {}
    per_count: dict[int, int] = {}
    for k, (R_cw, t_cw) in sorted(sol.poses.items()):
        frame = frame_list[k][0]
        R_gl, t_gl = opencv_to_opengl_world_from_camera(R_cw, t_cw)
        extrinsics.append(
            Extrinsic(
                frame=frame,
                translation=(float(t_gl[0]), float(t_gl[1]), float(t_gl[2])),
                rotation_euler_zxy_deg=rotation_matrix_to_euler_zxy_deg(R_gl),
                rotation_matrix=tuple(tuple(float(v) for v in row) for row in R_gl),  # type: ignore[arg-type]
            )
        )
        per_err[frame] = float(sol.frame_error_px.get(k, 0.0)) * scale
        per_count[frame] = int(sol.frame_inliers.get(k, 0))
    step = max(1, len(sol.points) // max_points)
    points = [
        ScenePoint(
            xyz=world_point_to_gl((float(p[0]), float(p[1]), float(p[2]))),
            color_rgb=(int(c[0]), int(c[1]), int(c[2])),
            reprojection_error_px=0.0,
            track_length=1,
        )
        for p, c in zip(sol.points[::step], sol.colors[::step], strict=True)
    ]
    registered = {e.frame for e in extrinsics}
    quality = _quality(per_err, per_count, len(registered), len(frame_list), intr, extrinsics)
    quality.warnings.insert(
        0,
        "Depth-anchored PnP camera (SfM found too little parallax): 3D from MoGe-3 depth of "
        f"frame {frame_list[sol.anchor][0]}, {fov_note or 'FOV estimated by MoGe-3'}; "
        "scale in metres is MoGe-3's estimate.",
    )
    if verdict is not None:
        _apply_verdict(quality, verdict, scale, len(frame_list), metric=True)
    return CameraSolve(
        shot=ShotRef(),
        backend=BackendInfo(
            name="pycolmap",
            version=_pycolmap_version(),
            matcher=str(params.get("matcher", "sequential")),
            sequential_overlap=int(params.get("sequential_overlap", 10)),
            camera_model="PINHOLE",
            sift_max_features=int(params.get("sift_max_features", 8192)),
            max_image_size=int(params.get("max_image_size", 1920)),
            mapper="depth_pnp",
            masked_labels=list(masked_labels or []),
        ),
        conventions=Conventions(
            units="metres",
            unit_note="Metric scale from MoGe-3 depth of one frame: an estimate, not a survey.",
        ),
        intrinsics=intr,
        extrinsics=extrinsics,
        unregistered_frames=sorted(fr for fr, _ in frame_list if fr not in registered),
        points=points,
        quality=quality,
    )


def _image_reprojection_errors(reconstruction: Any, im: Any) -> np.ndarray:
    """Pixel residuals (solve res) of every triangulated observation in `im`."""
    errs: list[float] = []
    for p2d in im.points2D:
        has = p2d.has_point3D
        if not (has() if callable(has) else has):
            continue
        pt = reconstruction.points3D[p2d.point3D_id]
        proj = im.project_point(pt.xyz)
        if proj is None:
            continue
        errs.append(
            float(
                np.hypot(
                    *(np.asarray(proj, dtype=np.float64) - np.asarray(p2d.xy, dtype=np.float64))
                )
            )
        )
    return np.asarray(errs, dtype=np.float64)


def _apply_verdict(
    quality: Quality, verdict: Any, scale: float, n_frames: int, *, metric: bool
) -> None:
    """Write a `pnp.PnPVerdict` into `quality`: the verdict line leads the
    warnings (hence the Nuke label), `verdict`, `heldout_error_px` (plate
    px = solve px x `scale`) and `nodal` feed the GUI light, `liveaov qc`
    and the splat gate. `metric`
    solves (PnP, metres from MoGe-3) state the travel in metres; an SfM
    solve has no scale, so only the parallax (travel over depth) is said."""
    held_plate: float | None = float(verdict.heldout_px) * scale
    assert held_plate is not None
    if math.isfinite(held_plate):
        held = f"held-out error {held_plate:.1f} px (plate res)"
    else:
        # The referee could not test the camera (its tracks were lost): no
        # number to report, and `assess` has already said FAILED.
        held, held_plate = "held-out error not measurable (tracks lost)", None
    line = (
        f"{verdict.verdict} CANDIDATE - {'NODAL' if verdict.nodal else 'translated'}: "
        f"{held}, {len(verdict.reliable)}/{n_frames} reliable frames, "
    )
    if metric:
        line += (
            f"travel {verdict.travel_m:.2f} m = {100 * verdict.parallax:.1f} % of the "
            f"{verdict.depth_m:.1f} m median depth"
        )
    else:
        line += (
            f"travel {100 * verdict.parallax:.1f} % of the median scene depth "
            "(KLT referee, held out frame by frame)"
        )
    if verdict.verdict == "FAILED":
        line += " - DO NOT USE"
    quality.warnings.insert(0, line)
    quality.verdict = verdict.verdict
    quality.heldout_error_px = held_plate
    quality.nodal = bool(verdict.nodal)


def _quality(
    per_frame_err: dict[int, float],
    per_frame_count: dict[int, int],
    registered: int,
    total: int,
    intr: Intrinsics,
    extrinsics: list[Extrinsic],
) -> Quality:
    aggregate = float(np.mean(list(per_frame_err.values()))) if per_frame_err else 0.0
    err_score = max(0.0, min(1.0, (2.0 - aggregate) / 1.5)) if per_frame_err else 0.0
    reg_ratio = registered / max(total, 1)
    warnings: list[str] = []
    if reg_ratio < 0.9:
        warnings.append(
            f"Only {registered}/{total} frames registered - the solve may have split "
            "(cut, whip pan, frames with no parallax)."
        )
    if aggregate > 1.5:
        warnings.append(
            f"Mean reprojection error {aggregate:.2f} px at plate resolution (>1.5 px): "
            "check for moving objects (run a SAM 3 matte in the same job to mask them) "
            "or lens distortion (camera_model SIMPLE_RADIAL)."
        )
    if per_frame_count and min(per_frame_count.values()) < 50:
        weakest = min(per_frame_count, key=lambda k: per_frame_count[k])
        warnings.append(f"Frame {weakest} has only {per_frame_count[weakest]} tracked points.")
    if len(extrinsics) >= 2:
        pos = np.array([e.translation for e in extrinsics])
        if np.linalg.norm(pos.max(0) - pos.min(0)) < 1e-6:
            warnings.append("Camera does not translate - nodal pan; depth/scale unreliable.")
    return Quality(
        per_frame_reprojection_error_px=per_frame_err,
        aggregate_reprojection_error_px=aggregate,
        tracked_points_count=per_frame_count,
        registered_frame_count=registered,
        total_frame_count=total,
        solve_confidence=round(0.7 * err_score + 0.3 * reg_ratio, 4),
        warnings=warnings,
    )


def write_points_ply(solve: CameraSolve, path: Path) -> None:
    """ASCII PLY of the sparse cloud, in the same Y-up world as the camera."""
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(solve.points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    for p in solve.points:
        x, y, z = p.xyz  # already in the Y-up GL world
        r, g, b = p.color_rgb
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_points_obj(solve: CameraSolve, path: Path) -> None:
    """The sparse cloud as an OBJ point cloud, for Nuke's ReadGeo2 (Nuke reads
    neither PLY nor USD points in the classic 3D system; verified on Nuke 16).
    The `p` elements matter: without them ReadGeo2 loads the vertices as
    nothing at all. Vertex colours ride along as `v x y z r g b`."""
    lines = ["# LiveActionAOV camera track point cloud (Y-up, same world as the camera)"]
    for p in solve.points:
        x, y, z = p.xyz
        r, g, b = p.color_rgb
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f} {r / 255:.4f} {g / 255:.4f} {b / 255:.4f}")
    n = len(solve.points)
    for i in range(0, n, 64):
        lines.append("p " + " ".join(str(k + 1) for k in range(i, min(i + 64, n))))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def world_point_to_gl(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """Same 180 deg X rotation of the world as the cameras get."""
    return float(xyz[0]), -float(xyz[1]), -float(xyz[2])


def _pycolmap_version() -> str:
    try:
        pycolmap: Any = importlib.import_module("pycolmap")  # no usable type stubs

        return str(getattr(pycolmap, "__version__", "unknown"))
    except ImportError:  # pragma: no cover
        return "unknown"


__all__ = [
    "PyColmapCameraPass",
    "build_solve_from_reconstruction",
    "opencv_to_opengl_world_from_camera",
    "rotation_matrix_to_euler_zxy_deg",
    "write_points_ply",
]
