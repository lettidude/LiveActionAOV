# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

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
  5. Sidecars in `<output>/camera_track/`: `<shot>.camera.json` (record),
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
import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    License,
    PassType,
    SidecarSpec,
    TemporalMode,
    UtilityPass,
)
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
    # SIFT wants display-referred contrast; scene-linear would crush shadows.
    input_colorspace = "srgb_display"

    produces_channels: list = []
    produces_sidecars = [
        SidecarSpec(name="camera", format="json"),
        SidecarSpec(name="camera_nk", format="nk"),
        SidecarSpec(name="camera_points", format="ply"),
    ]
    # Soft dep: SAM 3 masks exclude moving objects when SAM 3 is in the job.
    optional_artifacts = ["sam3_hard_masks"]
    # The solve itself, for downstream passes (Gaussian splat).
    provides_artifacts = ["camera_solve"]

    DEFAULT_PARAMS: dict[str, Any] = {
        "matcher": "sequential",  # "sequential" (video) | "exhaustive" (short / unordered)
        "sequential_overlap": 10,
        "mapper": "incremental",  # "incremental" | "global" (GLOMAP-style, pycolmap 4)
        "camera_model": "SIMPLE_PINHOLE",
        # Long edge of the images SIFT sees. PF0081 (2048 wide): 2048 ->
        # 0.86 px / 62 s per 32 frames, 1024 -> 0.6 px@1K / 21 s.
        "max_image_size": 2048,
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

    # --- shot entry point ---
    def run_shot(
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

        self._workdir = Path(tempfile.mkdtemp(prefix="laaov_colmap_"))
        image_dir = self._workdir / "images"
        image_dir.mkdir(parents=True)
        frame_list, plate_wh, solve_wh = self._export_frames(reader, frame_range, image_dir)
        if len(frame_list) < 3:
            raise RuntimeError(f"Camera track needs at least 3 frames; got {len(frame_list)}.")
        mask_dir = self._export_masks(frame_list, solve_wh)
        rec = self._solve_colmap(self._workdir, image_dir, mask_dir, frame_list)
        self._solve = build_solve_from_reconstruction(
            rec,
            frame_list=frame_list,
            plate_wh=plate_wh,
            params=self.params,
            masked_labels=self._masked_labels,
        )
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
        out_dir = Path(shot.output_dir or shot.folder) / "camera_track"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{shot.name}.camera.json"
        nk_path = out_dir / f"{shot.name}.camera.nk"
        ply_path = out_dir / f"{shot.name}.points.ply"
        obj_path = out_dir / f"{shot.name}.points.obj"
        json_path.write_text(self._solve.model_dump_json(indent=2), encoding="utf-8")
        nk_path.write_text(render_nuke_camera_script(self._solve), encoding="utf-8")
        write_points_ply(self._solve, ply_path)
        write_points_obj(self._solve, obj_path)
        paths: dict[str, Path] = {
            "camera": json_path,
            "camera_nk": nk_path,
            "camera_points": ply_path,
            "camera_points_obj": obj_path,
        }
        if self._sparse_dir is not None and self._sparse_dir.exists():
            model_dir = out_dir / f"{shot.name}_colmap"
            shutil.rmtree(model_dir, ignore_errors=True)
            shutil.copytree(self._sparse_dir, model_dir)
            paths["camera_colmap"] = model_dir
        if self._workdir is not None:
            if self.params.get("keep_workdir"):
                _log.info("COLMAP workdir kept at: %s", self._workdir)
            else:
                shutil.rmtree(self._workdir, ignore_errors=True)
        return paths

    def unload(self) -> None:
        # Safety net when emit_sidecars never ran (failed shot).
        if self._workdir is not None and not self.params.get("keep_workdir"):
            shutil.rmtree(self._workdir, ignore_errors=True)

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
        for f in range(frame_range[0], frame_range[1] + 1):
            arr, _attrs = reader.read_frame(f)
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
        pycolmap: Any = importlib.import_module("pycolmap")  # no usable type stubs

        db = workdir / "database.db"
        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = str(self.params["camera_model"]).upper()
        if mask_dir is not None:
            reader_opts.mask_path = str(mask_dir)
        ext = pycolmap.FeatureExtractionOptions()
        ext.sift.max_num_features = int(self.params["sift_max_features"])
        ext.sift.peak_threshold = float(self.params["sift_peak_threshold"])
        pycolmap.extract_features(
            database_path=db,
            image_path=image_dir,
            image_names=[n for _, n in frame_list],
            camera_mode=pycolmap.CameraMode.SINGLE,
            reader_options=reader_opts,
            extraction_options=ext,
        )
        if self.params["matcher"] == "exhaustive":
            pycolmap.match_exhaustive(db)
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = int(self.params["sequential_overlap"])
            pycolmap.match_sequential(db, pairing_options=pairing)
        sparse = workdir / "sparse"
        sparse.mkdir(exist_ok=True)
        if self.params.get("mapper") == "global":
            recs = pycolmap.global_mapping(
                database_path=db, image_path=image_dir, output_path=sparse
            )
        else:
            recs = pycolmap.incremental_mapping(
                database_path=db, image_path=image_dir, output_path=sparse
            )
        if not recs:
            raise RuntimeError(
                "Camera track: no reconstruction. Likely causes: no parallax "
                "(locked-off / pure pan), heavy motion blur, or a textureless plate."
            )
        key, rec = max(recs.items(), key=lambda kv: kv[1].num_reg_images())
        self._sparse_dir = sparse / str(key)
        if not self._sparse_dir.exists():  # some pycolmap builds write only the best model
            self._sparse_dir.mkdir(parents=True, exist_ok=True)
            rec.write(str(self._sparse_dir))
        return rec


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
) -> CameraSolve:
    """pycolmap `Reconstruction` -> `CameraSolve` (OpenGL poses, Nuke Eulers,
    plate-resolution intrinsics, per-frame reprojection error in plate px)."""
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
    return CameraSolve(
        shot=ShotRef(),
        backend=BackendInfo(
            name="pycolmap",
            version=_pycolmap_version(),
            matcher=str(params.get("matcher", "sequential")),
            sequential_overlap=int(params.get("sequential_overlap", 10)),
            camera_model=str(params.get("camera_model", "SIMPLE_PINHOLE")),
            sift_max_features=int(params.get("sift_max_features", 8192)),
            max_image_size=int(params.get("max_image_size", 2048)),
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
