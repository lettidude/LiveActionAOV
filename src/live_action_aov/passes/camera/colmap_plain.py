# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""A COLMAP result as plain data, for a process that must not load pycolmap.

On a Mac pycolmap and PyTorch each bring their own OpenMP runtime and one
process cannot hold both. With `colmap_isolate` on (the default on a Mac)
the process that runs the passes never imports pycolmap: the COLMAP child
(`colmap_worker`) saves what the camera pass reads, and this module reads
it back.

* `write_snapshot` / `read_snapshot`: the reconstruction. What comes back
  has the attributes `build_solve_from_reconstruction` and the referee
  read from a pycolmap `Reconstruction`, with the same numbers.
* `save_pairs` / `load_pairs`: the verified matches the rotation-only
  fallback reads from COLMAP's database.

No pycolmap and no torch import in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

Pairs = list[tuple[int, int, np.ndarray, np.ndarray]]


class _Rotation:
    def __init__(self, R: np.ndarray) -> None:
        self._R = R

    def matrix(self) -> np.ndarray:
        return self._R


class PlainPose:
    def __init__(self, R: np.ndarray, t: np.ndarray) -> None:
        self.rotation = _Rotation(R)
        self.translation = t


class PlainCamera:
    def __init__(self, model: str, params: list[float], width: int, height: int) -> None:
        self.model_name = model
        self.params = params
        self.width, self.height = width, height


class PlainImage:
    def __init__(
        self,
        name: str,
        camera_id: int,
        pose: PlainPose | None,
        reprojection_errors: np.ndarray,
    ) -> None:
        self.name, self.camera_id = name, camera_id
        self.has_pose = pose is not None
        self.cam_from_world = pose
        # Pixel residuals at solve resolution, computed next to COLMAP.
        self.reprojection_errors = reprojection_errors


class _Track:
    def __init__(self, n: int) -> None:
        self._n = n

    def length(self) -> int:
        return self._n


class PlainPoint3D:
    def __init__(
        self, xyz: np.ndarray, color: tuple[int, int, int] | None, error: float, track_len: int
    ) -> None:
        self.xyz, self.color, self.error = xyz, color, error
        self.track = _Track(track_len)


class PlainReconstruction:
    def __init__(self) -> None:
        self.cameras: dict[int, PlainCamera] = {}
        self.images: dict[int, PlainImage] = {}
        self.points3D: dict[int, PlainPoint3D] = {}

    def num_reg_images(self) -> int:
        return sum(1 for im in self.images.values() if im.has_pose)


def write_snapshot(reconstruction: Any, path: Path) -> None:
    """Save a pycolmap `Reconstruction` for `read_snapshot`. Runs in the
    COLMAP child; floats are stored exactly."""
    from live_action_aov.passes.camera.pycolmap_track import (
        _image_reprojection_errors,
        _is_registered,
        _pose,
    )

    cameras = [
        {
            "id": int(cid),
            "model": str(getattr(cam, "model_name", None) or getattr(cam, "model", "")),
            "params": [float(x) for x in cam.params],
            "width": int(cam.width),
            "height": int(cam.height),
        }
        for cid, cam in reconstruction.cameras.items()
    ]
    images: list[dict[str, Any]] = []
    errs: list[np.ndarray] = []
    at = 0
    for iid, im in reconstruction.images.items():
        entry: dict[str, Any] = {
            "id": int(iid),
            "name": str(im.name),
            "camera_id": int(im.camera_id),
            "pose": None,
            "errs": [at, 0],
        }
        if _is_registered(im):
            pose = _pose(im)
            R = np.asarray(pose.rotation.matrix(), dtype=np.float64)
            entry["pose"] = {
                "R": R.tolist(),
                # pycolmap hands out column-major matrices; numpy's products
                # round differently by layout, so the layout travels too.
                "order": "F" if R.flags.f_contiguous and not R.flags.c_contiguous else "C",
                "t": np.asarray(pose.translation, dtype=np.float64).reshape(3).tolist(),
            }
            e = _image_reprojection_errors(reconstruction, im)
            entry["errs"] = [at, int(e.size)]
            errs.append(e)
            at += int(e.size)
        images.append(entry)

    ids, xyz, rgb, has_rgb, perr, tlen = [], [], [], [], [], []
    for pid, pt in reconstruction.points3D.items():
        ids.append(int(pid))
        xyz.append(np.asarray(pt.xyz, dtype=np.float64).reshape(3))
        color = getattr(pt, "color", None)
        has_rgb.append(color is not None)
        rgb.append((0, 0, 0) if color is None else (int(color[0]), int(color[1]), int(color[2])))
        perr.append(float(getattr(pt, "error", 0.0)))
        track = getattr(pt, "track", None)
        tlen.append(int(track.length()) if track is not None and hasattr(track, "length") else 0)
    np.savez(
        path,
        meta=np.array(json.dumps({"cameras": cameras, "images": images})),
        errs=np.concatenate(errs) if errs else np.zeros(0, np.float64),
        point_ids=np.asarray(ids, dtype=np.int64),
        xyz=np.asarray(xyz, dtype=np.float64).reshape(-1, 3),
        rgb=np.asarray(rgb, dtype=np.int64).reshape(-1, 3),
        has_rgb=np.asarray(has_rgb, dtype=bool),
        point_error=np.asarray(perr, dtype=np.float64),
        track_len=np.asarray(tlen, dtype=np.int64),
    )


def read_snapshot(path: Path) -> PlainReconstruction:
    """What `write_snapshot` saved, as objects the camera pass can read."""
    rec = PlainReconstruction()
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        errs = np.asarray(z["errs"], dtype=np.float64)
        ids, xyz, rgb = z["point_ids"], z["xyz"], z["rgb"]
        has_rgb, perr, tlen = z["has_rgb"], z["point_error"], z["track_len"]
    for c in meta["cameras"]:
        rec.cameras[int(c["id"])] = PlainCamera(
            str(c["model"]), [float(x) for x in c["params"]], int(c["width"]), int(c["height"])
        )
    for im in meta["images"]:
        p = im["pose"]
        pose = None
        if p is not None:
            R = np.asarray(p["R"], dtype=np.float64)
            R = np.asfortranarray(R) if p.get("order") == "F" else np.ascontiguousarray(R)
            pose = PlainPose(R, np.asarray(p["t"], dtype=np.float64))
        start, count = im["errs"]
        rec.images[int(im["id"])] = PlainImage(
            str(im["name"]), int(im["camera_id"]), pose, errs[start : start + count]
        )
    for k in range(len(ids)):
        color = (int(rgb[k, 0]), int(rgb[k, 1]), int(rgb[k, 2])) if has_rgb[k] else None
        rec.points3D[int(ids[k])] = PlainPoint3D(xyz[k], color, float(perr[k]), int(tlen[k]))
    return rec


def save_pairs(path: Path, pairs: Pairs) -> None:
    """`read_verified_pairs`' result, for `load_pairs`."""
    counts = [len(xi) for _i, _j, xi, _xj in pairs]
    np.savez(
        path,
        ij=np.asarray([(i, j) for i, j, _a, _b in pairs], dtype=np.int64).reshape(-1, 2),
        counts=np.asarray(counts, dtype=np.int64),
        xi=np.concatenate([p[2] for p in pairs]) if pairs else np.zeros((0, 2), np.float64),
        xj=np.concatenate([p[3] for p in pairs]) if pairs else np.zeros((0, 2), np.float64),
    )


def load_pairs(path: Path) -> Pairs:
    with np.load(path, allow_pickle=False) as z:
        ij, counts, xi, xj = z["ij"], z["counts"], z["xi"], z["xj"]
    out: Pairs = []
    at = 0
    for (i, j), n in zip(ij, counts, strict=True):
        out.append((int(i), int(j), xi[at : at + n], xj[at : at + n]))
        at += int(n)
    return out


__all__ = [
    "PlainReconstruction",
    "load_pairs",
    "read_snapshot",
    "save_pairs",
    "write_snapshot",
]
