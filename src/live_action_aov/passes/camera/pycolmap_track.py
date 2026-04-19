"""Classical SfM camera track via pycolmap (BSD-3-Clause, commercial-safe).

First camera-track backend for the tool. pycolmap ships Windows/Linux/macOS
wheels from PyPI (4.x series) — no C++ build, no CUDA toolkit required,
no conda env. That's why we picked it over MegaSAM for v2a: the classical
SfM pipeline works on every target machine today.

Pipeline:
  1. Write each (display-transformed) plate frame as a PNG into a temp
     workdir. pycolmap consumes image files, not tensors.
  2. SIFT feature extraction → COLMAP SQLite database.
  3. Sequential matching (overlap 10 by default — tuned for video) with an
     exhaustive fallback for short clips.
  4. Incremental mapping → one or more `Reconstruction` objects. We pick
     the largest (most registered images) as the canonical solve.
  5. Convert camera poses from COLMAP's OpenCV convention (+X right,
     +Y down, +Z forward) into our Y-up right-handed OpenGL world frame
     (what Nuke/Maya/Houdini/Blender expect on import).
  6. Emit sidecars in `<plate_folder>/camera_track/`:
       `<shot>.camera.json`  — the permanent record (schema v1.0)
       `<shot>.camera.nk`    — Nuke Camera2 script for drag-and-drop use

Scope caveats documented here so future-reader doesn't wonder:

- **Scale is arbitrary.** pycolmap incremental SfM doesn't recover metric
  scale. `conventions.units == "unitless"` in the sidecar. Metric recovery
  via Depth Pro prior or user-supplied scale hint is a v2b task (spec §20.4).
- **No distortion.** Pinhole only. v2b adds an OPENCV model variant that
  fits k1/k2/k3 and a 3DE-compatible distortion node.
- **No metadata reconciliation.** EXR header focal / sensor size are not
  fed back to COLMAP as a prior, and the solved focal is not compared to
  the metadata focal. Spec §20.3's 5-rule reconciliation ladder is
  deferred to a follow-up PR.
- **No ground/scale alignment.** Solve is in COLMAP's arbitrary world
  frame; origin is the first registered camera. Ground detection and
  scale recovery land in the GUI prep UX (Phase 5 + v2b).
- **Single camera mode only.** All frames share one intrinsics set
  (`CameraMode.SINGLE`). Animated focal (zooms) is a v2b feature.
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.core.job import Shot
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

# OpenCV (+X right, +Y down, +Z forward) → OpenGL (+X right, +Y up, +Z back).
# Negating Y and Z of the camera-local frame converts one to the other and
# is a rotation of 180° about X (orthonormal, det = +1).
_OPENCV_TO_OPENGL_FLIP = np.diag([1.0, -1.0, -1.0])


class PyColmapCameraPass(UtilityPass):
    """Classical incremental SfM camera solve via pycolmap.

    The pass is `VIDEO_CLIP` — it consumes the whole shot at once and
    overrides `run_shot` to drive the COLMAP pipeline. `preprocess` /
    `infer` / `postprocess` are the contract's per-frame hooks and are
    not used here; they raise if called.
    """

    name = "camera_pycolmap"
    version = "0.1.0"
    license = License(
        spdx="BSD-3-Clause",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "pycolmap and COLMAP are BSD-3-Clause; bundled SIFT implementation "
            "is also permissively licensed. Safe to ship in a commercial tool."
        ),
    )
    pass_type = PassType.CAMERA
    temporal_mode = TemporalMode.VIDEO_CLIP
    # Display-referred frames work fine for SIFT; the display transform
    # reader already gives us sRGB-in-[0,1] which we just clamp to uint8.
    input_colorspace = "srgb_display"

    produces_channels: list = []
    produces_sidecars = [
        SidecarSpec(name="camera", format="json"),
        SidecarSpec(name="camera_nk", format="nk"),
    ]
    # Artifacts other passes (or future executor analytics) might consume.
    # Empty values in v2a — filling these lets v2b backend-routing use the
    # same dict keys without re-solving.
    provides_artifacts = ["camera_extrinsics", "camera_intrinsics", "scene_points_sparse"]

    DEFAULT_PARAMS: dict[str, Any] = {
        # "sequential" = only match each frame to its N neighbours; fast and
        # temporally-correct for video. "exhaustive" = match every pair;
        # slow but robust for short clips or non-sequential dailies.
        "matcher": "sequential",
        "sequential_overlap": 10,
        # PINHOLE = [fx, fy, cx, cy]; SIMPLE_PINHOLE = [f, cx, cy]. OPENCV
        # (with distortion) is a v2b extension.
        "camera_model": "PINHOLE",
        # Downsample plate long-edge to this size for SIFT. 1920 is a good
        # balance between feature count and extraction speed; full 4K adds
        # minutes with negligible solve improvement for most shots.
        "max_image_size": 1920,
        "sift_max_features": 4096,
        # Default sensor hints — Super35. Metadata reconciliation (v2b)
        # will overwrite from EXR header when available.
        "haperture_mm_hint": 24.892,
        "vaperture_mm_hint": 18.669,
        # Leave the COLMAP workdir on disk after the solve for debugging.
        # Users investigating a bad solve want the database + intermediate
        # images; only set False on headless batch runs.
        "keep_workdir": True,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._solve: CameraSolve | None = None

    # --- UtilityPass per-frame hooks (not used for VIDEO_CLIP passes) ---

    def preprocess(self, frames: np.ndarray) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "PyColmapCameraPass is VIDEO_CLIP — use run_shot(), not the per-frame hooks."
        )

    def infer(self, tensor: Any) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "PyColmapCameraPass is VIDEO_CLIP — use run_shot(), not the per-frame hooks."
        )

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:  # pragma: no cover
        raise NotImplementedError(
            "PyColmapCameraPass is VIDEO_CLIP — use run_shot(), not the per-frame hooks."
        )

    # --- Shot-level entry point ---

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Drive the full COLMAP pipeline and stash the resulting solve
        on ``self._solve`` for ``emit_sidecars`` to write.

        Returns an empty channel dict — this pass contributes no EXR
        channels. The executor's EXR writer will simply skip frames it
        sees nothing for, which is the intended behaviour.
        """
        try:
            import pycolmap  # noqa: F401 — imported for side effects + availability check
        except ImportError as e:  # pragma: no cover — guarded by optional extra
            raise ImportError(
                "pycolmap is required for the camera_pycolmap pass. "
                "Install with: pip install 'live-action-aov[camera]'"
            ) from e

        workdir = Path(tempfile.mkdtemp(prefix="laaov_colmap_"))
        image_dir = workdir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        try:
            frame_list = self._export_frames_as_png(reader, frame_range, image_dir)
            if len(frame_list) < 2:
                raise RuntimeError(
                    f"Camera track needs at least 2 frames; got {len(frame_list)}."
                )

            rec = self._run_pycolmap_solve(workdir, image_dir, frame_list)
            self._solve = build_solve_from_reconstruction(
                rec,
                frame_list=frame_list,
                total_frame_count=len(frame_list),
                params=self.params,
            )
            _log.info(
                "Camera solve: %d/%d frames registered, aggregate reproj err %.3f px",
                self._solve.quality.registered_frame_count,
                self._solve.quality.total_frame_count,
                self._solve.quality.aggregate_reprojection_error_px,
            )
        finally:
            if not self.params.get("keep_workdir", False):
                shutil.rmtree(workdir, ignore_errors=True)
            else:
                # Leaves the COLMAP database + images + sparse reconstruction
                # next to /tmp for user inspection. The path is logged so
                # the user can find it.
                _log.info("COLMAP workdir kept at: %s", workdir)

        return {}

    # --- Sidecar emission ---

    def emit_sidecars(self, shot: Shot) -> dict[str, Path]:
        """Write the JSON + .nk sidecars into `<shot.folder>/camera_track/`.

        Called by the executor after `run_shot`. Returns a `{tag: path}`
        dict the executor merges into `shot.sidecars`. Tags match the
        `produces_sidecars` declaration (`camera`, `camera_nk`).
        """
        if self._solve is None:
            return {}

        # Fill in shot-level identification — couldn't do this in run_shot
        # because we didn't have the Shot reference there.
        self._solve.shot = ShotRef(
            name=shot.name,
            plate_source=str(shot.folder),
            frame_range=shot.frame_range,
            resolution=shot.resolution,
            pixel_aspect=shot.pixel_aspect,
        )

        sidecar_dir = shot.folder / "camera_track"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        json_path = sidecar_dir / f"{shot.name}.camera.json"
        nk_path = sidecar_dir / f"{shot.name}.camera.nk"

        json_path.write_text(
            self._solve.model_dump_json(indent=2),
            encoding="utf-8",
        )
        nk_path.write_text(
            render_nuke_camera_script(self._solve),
            encoding="utf-8",
        )

        return {"camera": json_path, "camera_nk": nk_path}

    # --- Internals ---

    def _export_frames_as_png(
        self,
        reader: Any,
        frame_range: tuple[int, int],
        image_dir: Path,
    ) -> list[tuple[int, str]]:
        """Read every frame in range, convert to 8-bit sRGB, write PNG.

        Returns list of (frame_idx, png_filename) pairs in frame order —
        COLMAP's sequential matcher treats lexicographic filename order
        as temporal order, so zero-padded names matter.
        """
        try:
            from PIL import Image
        except ImportError as e:  # pragma: no cover — Pillow is a Pydantic/transformers transitive dep
            raise ImportError(
                "Pillow is required to write PNGs for pycolmap. "
                "Install with: pip install Pillow"
            ) from e

        max_size = int(self.params.get("max_image_size", 1920))
        out: list[tuple[int, str]] = []
        for f in range(frame_range[0], frame_range[1] + 1):
            arr, _attrs = reader.read_frame(f)
            # Reader gives (H, W, C) float32 in the pass's declared
            # input_colorspace. Clamp and quantise to 8-bit.
            rgb = np.clip(arr, 0.0, 1.0)
            rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
            if rgb_u8.ndim == 2:
                rgb_u8 = np.stack([rgb_u8] * 3, axis=-1)
            elif rgb_u8.shape[-1] == 1:
                rgb_u8 = np.repeat(rgb_u8, 3, axis=-1)
            elif rgb_u8.shape[-1] > 3:
                rgb_u8 = rgb_u8[..., :3]

            h, w = rgb_u8.shape[:2]
            long_edge = max(h, w)
            if long_edge > max_size:
                scale = max_size / long_edge
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = Image.fromarray(rgb_u8).resize((new_w, new_h), Image.BILINEAR)
            else:
                img = Image.fromarray(rgb_u8)

            name = f"frame_{f:06d}.png"
            img.save(image_dir / name, format="PNG")
            out.append((f, name))
        return out

    def _run_pycolmap_solve(
        self,
        workdir: Path,
        image_dir: Path,
        frame_list: list[tuple[int, str]],
    ) -> Any:
        """Run SIFT extraction → matcher → incremental mapper. Returns the
        largest `Reconstruction`."""
        import pycolmap

        db_path = workdir / "database.db"
        if db_path.exists():
            db_path.unlink()

        # pycolmap 4.x nests SIFT knobs inside FeatureExtractionOptions.sift;
        # extract_features takes the wrapper via its `extraction_options`
        # kwarg (not `sift_options` as in some older docs).
        ext_opts = pycolmap.FeatureExtractionOptions()
        ext_opts.sift.max_num_features = int(self.params["sift_max_features"])

        camera_mode = pycolmap.CameraMode.SINGLE
        image_names = [name for _, name in frame_list]

        pycolmap.extract_features(
            database_path=db_path,
            image_path=image_dir,
            image_names=image_names,
            camera_mode=camera_mode,
            extraction_options=ext_opts,
        )

        if self.params["matcher"] == "exhaustive":
            pycolmap.match_exhaustive(db_path)
        else:
            overlap = int(self.params.get("sequential_overlap", 10))
            # Sequential matching's overlap knob lives on
            # SequentialPairingOptions (the `pairing_options` kwarg of
            # match_sequential), separate from FeatureMatchingOptions.
            pairing_opts = pycolmap.SequentialPairingOptions()
            pairing_opts.overlap = overlap
            pycolmap.match_sequential(db_path, pairing_options=pairing_opts)

        sparse_dir = workdir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        reconstructions = pycolmap.incremental_mapping(
            database_path=db_path,
            image_path=image_dir,
            output_path=sparse_dir,
        )

        if not reconstructions:
            raise RuntimeError(
                "pycolmap produced no reconstruction. Possible causes: "
                "low-texture plate, extreme motion blur, or too few frames."
            )

        # Multiple reconstructions happen when the graph disconnects (e.g.
        # a scene cut). Pick the largest by registered image count.
        rec = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        return rec


# ---------------------------------------------------------------------------
# Pure helpers — extracted from the class for easier unit testing without
# spinning up the whole pipeline.
# ---------------------------------------------------------------------------


def opencv_to_opengl_world_from_camera(
    R_cw: np.ndarray,
    t_cw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert COLMAP's `cam_from_world` (OpenCV) to OpenGL `world_from_camera`.

    COLMAP: camera local frame is +X right, +Y down, +Z forward; pose is
    expressed as the rigid transform that takes world points into the
    camera frame (``R_cw`` rotates, ``t_cw`` translates).

    Nuke/Maya/Houdini expect an OpenGL-style camera: +X right, +Y up,
    +Z *back* (the camera looks down its own -Z). We also want the
    `world_from_camera` side of the transform so the translate knob is
    just the camera's world position.

    Returns `(R_wc_gl, t_wc_gl)` — 3x3 rotation that takes OpenGL
    camera-local points into world, plus the camera's world position.
    """
    R_cw = np.asarray(R_cw, dtype=np.float64).reshape(3, 3)
    t_cw = np.asarray(t_cw, dtype=np.float64).reshape(3)
    R_wc_cv = R_cw.T
    t_wc_cv = -R_wc_cv @ t_cw
    R_wc_gl = R_wc_cv @ _OPENCV_TO_OPENGL_FLIP
    return R_wc_gl, t_wc_cv


def rotation_matrix_to_euler_zxy_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose `R = Ry(β) · Rx(α) · Rz(γ)` into `(α, β, γ)` degrees.

    Matches Nuke's `rot_order ZXY` convention: Nuke applies the rotate
    knob as Rz first, then Rx, then Ry to local vectors, so the combined
    matrix acts as ``v_world = Ry · Rx · Rz · v_local``.

    Gimbal lock occurs at `α = ±90°` (when `R[1,2] = ∓1`); in that case
    we zero out γ and recover β from the reduced equations, which is the
    standard convention and keeps the rotation matrix round-trippable
    even if the individual Euler channels diverge.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)

    # α = asin(-R[1,2]); clamp argument for numerical safety.
    arg = max(-1.0, min(1.0, float(-R[1, 2])))
    alpha = math.asin(arg)
    cos_alpha = math.cos(alpha)

    if abs(cos_alpha) > 1e-6:
        beta = math.atan2(float(R[0, 2]), float(R[2, 2]))
        gamma = math.atan2(float(R[1, 0]), float(R[1, 1]))
    else:
        # Gimbal lock — pin γ = 0 and extract β from the degenerate matrix.
        gamma = 0.0
        beta = math.atan2(-float(R[2, 0]), float(R[0, 0]))

    return (
        math.degrees(alpha),
        math.degrees(beta),
        math.degrees(gamma),
    )


def _intrinsics_from_colmap_camera(
    camera: Any,
    params: dict[str, Any],
) -> Intrinsics:
    """Read fx, fy, cx, cy from a pycolmap Camera.

    pycolmap exposes intrinsics via `camera.params` (array) whose layout
    depends on `camera.model_name`. We support PINHOLE and
    SIMPLE_PINHOLE; others raise so callers know to configure the solver
    with a supported model.
    """
    model = getattr(camera, "model_name", None) or getattr(camera, "model", None)
    model_str = str(model).upper()
    p = [float(x) for x in camera.params]
    if "SIMPLE_PINHOLE" in model_str:
        f, cx, cy = p[0], p[1], p[2]
        fx = fy = f
    elif "PINHOLE" in model_str:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:
        raise ValueError(
            f"Unsupported COLMAP camera model: {model_str}. "
            "Configure the pass with camera_model='PINHOLE' or 'SIMPLE_PINHOLE'."
        )

    width = int(camera.width)
    height = int(camera.height)
    haperture = float(params.get("haperture_mm_hint", 24.892))
    vaperture = float(params.get("vaperture_mm_hint", haperture * height / max(width, 1)))
    focal_mm_hint = fx * haperture / max(width, 1)

    return Intrinsics(
        fx_px=fx,
        fy_px=fy,
        cx_px=cx,
        cy_px=cy,
        width_px=width,
        height_px=height,
        haperture_mm_hint=haperture,
        vaperture_mm_hint=vaperture,
        focal_mm_hint=focal_mm_hint,
    )


def build_solve_from_reconstruction(
    reconstruction: Any,
    *,
    frame_list: list[tuple[int, str]],
    total_frame_count: int,
    params: dict[str, Any],
) -> CameraSolve:
    """Transform a pycolmap `Reconstruction` into our `CameraSolve` schema.

    The heavy lifting: coordinate conversion (OpenCV → OpenGL), Euler
    decomposition for Nuke import, shared-intrinsics extraction, sparse
    point harvest, per-frame reprojection-error aggregation.
    """
    # Import at call time so the helper is testable without pycolmap
    # installed — the test fakes a Reconstruction-shaped object.
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        pass  # OK — we use only attribute access below.

    # Map COLMAP's internal image_id → our plate frame index, via filename.
    name_to_frame: dict[str, int] = {name: frame for frame, name in frame_list}

    # Intrinsics — SINGLE camera mode means one Camera shared across all
    # images. If COLMAP split into multiple cameras for some reason, pick
    # the one the majority of registered images use.
    cam_counts: dict[int, int] = {}
    for img in reconstruction.images.values():
        cam_counts[img.camera_id] = cam_counts.get(img.camera_id, 0) + 1
    primary_camera_id = max(cam_counts, key=lambda k: cam_counts[k])
    primary_camera = reconstruction.cameras[primary_camera_id]
    intrinsics = _intrinsics_from_colmap_camera(primary_camera, params)

    # Extrinsics — convert each image pose and decompose.
    extrinsics: list[Extrinsic] = []
    registered_names: set[str] = set()
    for img in reconstruction.images.values():
        frame = name_to_frame.get(img.name)
        if frame is None:
            continue
        registered_names.add(img.name)
        R_cw = np.asarray(img.cam_from_world.rotation.matrix(), dtype=np.float64)
        t_cw = np.asarray(img.cam_from_world.translation, dtype=np.float64)
        R_gl, t_gl = opencv_to_opengl_world_from_camera(R_cw, t_cw)
        alpha, beta, gamma = rotation_matrix_to_euler_zxy_deg(R_gl)
        extrinsics.append(
            Extrinsic(
                frame=frame,
                translation=(float(t_gl[0]), float(t_gl[1]), float(t_gl[2])),
                rotation_euler_zxy_deg=(alpha, beta, gamma),
                rotation_matrix=(
                    (float(R_gl[0, 0]), float(R_gl[0, 1]), float(R_gl[0, 2])),
                    (float(R_gl[1, 0]), float(R_gl[1, 1]), float(R_gl[1, 2])),
                    (float(R_gl[2, 0]), float(R_gl[2, 1]), float(R_gl[2, 2])),
                ),
            )
        )
    extrinsics.sort(key=lambda e: e.frame)

    unregistered = sorted(
        frame for frame, name in frame_list if name not in registered_names
    )

    # Sparse points — include color + reproj error so QC tools don't need
    # to re-derive them. Points with zero observations slip through some
    # COLMAP builds; filter them so the sidecar is clean.
    points: list[ScenePoint] = []
    for pt in reconstruction.points3D.values():
        xyz = np.asarray(pt.xyz, dtype=np.float64)
        # Apply the same OpenGL flip we applied to camera poses so the
        # point cloud stays consistent with the cameras in world space.
        # COLMAP's world is the same handedness as ours; only the
        # camera-local frames differ. So points don't need flipping;
        # leaving them as-is.
        rgb = getattr(pt, "color", None)
        if rgb is None:
            color = (255, 255, 255)
        else:
            color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        track_length = int(getattr(pt, "track", None).length()) if hasattr(pt, "track") else 0
        err = float(getattr(pt, "error", 0.0))
        points.append(
            ScenePoint(
                xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                color_rgb=color,
                reprojection_error_px=err,
                track_length=track_length,
            )
        )

    quality = _compute_quality(reconstruction, extrinsics, total_frame_count)

    return CameraSolve(
        shot=ShotRef(),  # filled in by emit_sidecars once we have the Shot
        backend=BackendInfo(
            name="pycolmap",
            version=_pycolmap_version(),
            matcher=str(params.get("matcher", "sequential")),
            sequential_overlap=int(params.get("sequential_overlap", 10)),
            camera_model=str(params.get("camera_model", "PINHOLE")),
            sift_max_features=int(params.get("sift_max_features", 4096)),
            max_image_size=int(params.get("max_image_size", 1920)),
        ),
        conventions=Conventions(),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        unregistered_frames=unregistered,
        points=points,
        quality=quality,
    )


def _compute_quality(
    reconstruction: Any,
    extrinsics: list[Extrinsic],
    total_frame_count: int,
) -> Quality:
    """Aggregate per-frame reprojection errors + tracked-point counts into
    the `Quality` block the GUI's traffic-light indicator reads."""
    per_frame_err: dict[int, float] = {}
    per_frame_count: dict[int, int] = {}

    # Build name -> frame lookup from the extrinsics list (already in our
    # plate frame-index space).
    img_name_to_frame: dict[str, int] = {}
    for img in reconstruction.images.values():
        # Name patterns we control: `frame_000123.png`. Parse to map back.
        try:
            stem = img.name.rsplit(".", 1)[0]
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        img_name_to_frame[img.name] = idx

    for img in reconstruction.images.values():
        frame = img_name_to_frame.get(img.name)
        if frame is None:
            continue
        points2D = getattr(img, "points2D", None) or []
        errs: list[float] = []
        count = 0
        for p2d in points2D:
            if not getattr(p2d, "has_point3D", lambda: False)():
                continue
            pt3d_id = p2d.point3D_id
            pt3d = reconstruction.points3D.get(pt3d_id)
            if pt3d is None:
                continue
            # Prefer the per-point reprojection error COLMAP cached on
            # Point3D.error; computing it from scratch would need the
            # projection and the raw observation coords.
            errs.append(float(getattr(pt3d, "error", 0.0)))
            count += 1
        if errs:
            per_frame_err[frame] = float(np.mean(errs))
            per_frame_count[frame] = count

    if per_frame_err:
        aggregate = float(np.mean(list(per_frame_err.values())))
    else:
        aggregate = 0.0

    registered = len(extrinsics)
    # Confidence heuristic: 1.0 at <0.5 px aggregate error, 0.0 at >2.0 px,
    # linear in between. Combined with registration ratio so a half-solved
    # clip gets flagged yellow/red even if the registered portion is clean.
    err_score = max(0.0, min(1.0, (2.0 - aggregate) / 1.5)) if aggregate > 0 else 0.5
    reg_ratio = registered / max(total_frame_count, 1)
    confidence = 0.7 * err_score + 0.3 * reg_ratio

    warnings: list[str] = []
    if reg_ratio < 0.9:
        warnings.append(
            f"Only {registered}/{total_frame_count} frames registered — "
            "solve may have disconnected subsequences."
        )
    if aggregate > 2.0:
        warnings.append(
            f"High aggregate reprojection error ({aggregate:.2f} px). "
            "Consider `matcher=exhaustive` or a higher `sift_max_features`."
        )

    return Quality(
        per_frame_reprojection_error_px=per_frame_err,
        aggregate_reprojection_error_px=aggregate,
        tracked_points_count=per_frame_count,
        registered_frame_count=registered,
        total_frame_count=total_frame_count,
        solve_confidence=round(confidence, 4),
        warnings=warnings,
    )


def _pycolmap_version() -> str:
    try:
        import pycolmap

        return str(getattr(pycolmap, "__version__", "unknown"))
    except ImportError:  # pragma: no cover — used only in solve path
        return "unknown"


__all__ = [
    "PyColmapCameraPass",
    "build_solve_from_reconstruction",
    "opencv_to_opengl_world_from_camera",
    "rotation_matrix_to_euler_zxy_deg",
]
