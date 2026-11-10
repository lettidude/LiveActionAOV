# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)
"""A point cloud an artist can read in a 3D viewport.

The camera solve's own cloud is a few thousand feature points: enough to
check a track, not enough to recognise the set. This module builds the cloud
that is written for people:

- **dense** when the run has a depth pass: a handful of keyframes are pushed
  back into the camera's world through the solved camera. Each keyframe's
  depth is brought to the solve's units with the feature points seen in that
  frame (their solved distance against the depth under them), tracked objects
  and depth edges are left out, and a keyframe only adds what the earlier
  ones did not already cover, so a surface is drawn once;
- **sparse** otherwise: the feature points, without the outliers, coloured
  from the plate.

Colours come from the plate as the models saw it, lifted so a dark plate
still reads on a grey viewport.

`write_cloud_obj` writes the form Nuke's classic 3D system shows well: one
small quad per point (a vertex-only OBJ draws as 1 px dots and does not
render), with a colour atlas PNG next to it that the quads' UVs point into.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from live_action_aov.io.camera_schema import CameraSolve

#: frame -> (H, W, 3) display-referred RGB, any resolution.
PlateReader = Callable[[int], np.ndarray]
#: frame -> {"Z": ..., "mask.<name>": ..., "depth.confidence": ...} at plate resolution.
PassReader = Callable[[int], dict[str, np.ndarray]]

_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


@dataclass
class DisplayCloud:
    xyz: np.ndarray  # (N, 3) float32, the camera's Y-up world
    rgb: np.ndarray  # (N, 3) float32 in [0, 1], display-referred
    size: np.ndarray  # (N,) float32, world-space width of each point
    dense: bool
    note: str

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


def _project(
    xyz: np.ndarray, rot: np.ndarray, trans: np.ndarray, k: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> (u, v, distance along the view) for a GL camera
    (`rot` = world_from_camera, the camera looks down -Z)."""
    fx, fy, cx, cy = k
    cam = (xyz - trans) @ rot
    z = -cam[:, 2]
    safe = np.maximum(z, 1e-9)
    return cx + fx * cam[:, 0] / safe, cy - fy * cam[:, 1] / safe, z


def _sample(img: np.ndarray, u: np.ndarray, v: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest pixel of `img` at plate-pixel coordinates (the image may be a proxy)."""
    ih, iw = img.shape[0], img.shape[1]
    x = np.clip((u / w * iw).astype(np.int64), 0, iw - 1)
    y = np.clip((v / h * ih).astype(np.int64), 0, ih - 1)
    return np.asarray(img[y, x, :3], np.float32)


def _lift(rgb: np.ndarray) -> np.ndarray:
    """Gain so the bright end of the cloud sits around 0.55: a night plate
    stays readable, a normal one is left alone."""
    if rgb.size == 0:
        return rgb
    p90 = float(np.percentile(np.clip(rgb, 0, None) @ _LUMA, 90))
    gain = float(np.clip(0.55 / max(p90, 1e-4), 1.0, 6.0))
    return np.clip(rgb * gain, 0.0, 1.0).astype(np.float32)


def build_display_cloud(
    solve: CameraSolve,
    read_plate: PlateReader | None = None,
    read_passes: PassReader | None = None,
    *,
    max_points: int = 60000,
    keyframes: int = 6,
    seed: int = 0,
) -> DisplayCloud | None:
    """The cloud to show (see the module docstring). None without a solve."""
    if not solve.extrinsics or not solve.points:
        return None
    intr = solve.intrinsics
    w, h = int(intr.width_px), int(intr.height_px)
    k = (float(intr.fx_px), float(intr.fy_px), float(intr.cx_px), float(intr.cy_px))
    ext = {
        e.frame: (np.asarray(e.rotation_matrix, np.float64), np.asarray(e.translation, np.float64))
        for e in solve.extrinsics
    }
    frames = sorted(ext)
    cam_pos = np.array([ext[f][1] for f in frames])

    # --- the feature points, without the outliers ---
    pts = np.array([p.xyz for p in solve.points], np.float64)
    err = np.array([p.reprojection_error_px for p in solve.points], np.float64)
    seen = np.array([p.track_length for p in solve.points], np.int64)
    near = cam_pos[:: max(1, len(frames) // 8)]
    dist = np.linalg.norm(pts[:, None] - near[None], axis=2).min(1)
    ok = (err <= max(2.0, 3.0 * float(np.median(err)))) & (dist <= np.percentile(dist, 98))
    if int((seen >= 3).sum()) >= 50:  # solvers that do not report tracks leave it at 0
        ok &= seen >= 3
    sparse_rgb = np.array([p.color_rgb for p in solve.points], np.float32) / 255.0
    if int(ok.sum()) >= 50:
        pts, dist, sparse_rgb = pts[ok], dist[ok], sparse_rgb[ok]
    if read_plate is not None:
        sparse_rgb = np.full((len(pts), 3), 0.5, np.float32)
        todo = np.ones(len(pts), bool)
        for f in frames[:: max(1, len(frames) // 10)]:
            if not todo.any():
                break
            u, v, z = _project(pts, *ext[f], k)
            m = todo & (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if m.any():
                sparse_rgb[m] = _sample(read_plate(f), u[m], v[m], w, h)
                todo &= ~m

    # --- dense: keyframes of the depth pass, pushed back through the camera ---
    rng = np.random.default_rng(seed)
    stride = max(2, int(np.ceil(np.sqrt(w * h / (0.6 * max(max_points, 1))))))
    gh, gw = -(-h // stride), -(-w // stride)
    mid = len(frames) // 2
    step = max(1, len(frames) // max(keyframes, 1))
    order = sorted(frames[::step][:keyframes], key=lambda f: abs(frames.index(f) - mid))
    d_xyz: list[np.ndarray] = []
    d_rgb: list[np.ndarray] = []
    d_size: list[np.ndarray] = []
    used = 0
    for f in order if (read_passes is not None and read_plate is not None) else []:
        assert read_passes is not None and read_plate is not None
        ch = read_passes(f)
        depth = ch.get("Z")
        if depth is None or depth.shape[:2] != (h, w):
            continue
        depth = np.asarray(depth, np.float64)
        rot, trans = ext[f]
        u, v, z = _project(pts, rot, trans, k)
        m = (z > 0) & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        under = depth[v[m].astype(np.int64), u[m].astype(np.int64)]
        good = under > 1e-4
        if int(good.sum()) < 30:
            continue  # nothing to tie this frame's depth to the solve
        ratio = z[m][good] / under[good]
        scale = float(np.median(ratio))
        if float(np.median(np.abs(ratio / scale - 1.0))) > 0.12:
            continue  # this frame's depth does not agree with the solve
        gy, gx = np.gradient(depth)
        keep = (depth > 1e-4) & (np.hypot(gx, gy) / np.maximum(depth, 1e-4) < 0.03)
        # Tracked objects move: the set only. Their masks are grown by about
        # 1.2% of the width first, or the hair and the motion blur just outside
        # a mask come back as a ghost outline at every keyframe.
        objects = np.zeros((gh, gw), bool)
        for name, arr in ch.items():
            if name.startswith(("mask.", "matte.")) and arr.shape[:2] == (h, w):
                padded = np.zeros((gh * stride, gw * stride), bool)
                padded[:h, :w] = arr > 0.05
                objects |= padded.reshape(gh, stride, gw, stride).any(axis=(1, 3))
        for _ in range(max(1, round(0.012 * w / stride))):
            grown = objects.copy()
            grown[1:] |= objects[:-1]
            grown[:-1] |= objects[1:]
            grown[:, 1:] |= objects[:, :-1]
            grown[:, :-1] |= objects[:, 1:]
            objects = grown
        conf = ch.get("depth.confidence")
        if conf is not None and conf.shape[:2] == (h, w):
            keep &= conf > 0.5
        covered = np.zeros((gh, gw), bool)
        if d_xyz:
            pu, pv, pz = _project(np.concatenate(d_xyz), rot, trans, k)
            vis = (pz > 0) & (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
            covered[(pv[vis] // stride).astype(np.int64), (pu[vis] // stride).astype(np.int64)] = (
                True
            )
            grown = covered.copy()
            grown[1:] |= covered[:-1]
            grown[:-1] |= covered[1:]
            grown[:, 1:] |= covered[:, :-1]
            grown[:, :-1] |= covered[:, 1:]
            covered = grown
        cy_, cx_ = np.nonzero(~covered & ~objects)
        # One pixel per free cell, jittered around its middle: a regular grid
        # reads as a moire, a fully random one leaves holes.
        lo, span = stride // 4, stride // 2 + 1
        vv = np.minimum(cy_ * stride + lo + rng.integers(0, span, len(cy_)), h - 1)
        uu = np.minimum(cx_ * stride + lo + rng.integers(0, span, len(cx_)), w - 1)
        sel = keep[vv, uu]
        vv, uu = vv[sel], uu[sel]
        if len(vv) == 0:
            continue
        zz = depth[vv, uu] * scale
        cam = np.stack([(uu - k[2]) / k[0] * zz, -(vv - k[3]) / k[1] * zz, -zz], -1)
        d_xyz.append(cam @ rot.T + trans)
        d_rgb.append(_sample(read_plate(f), uu.astype(np.float64), vv.astype(np.float64), w, h))
        d_size.append(zz * (stride * 1.4) / k[0])  # overlap a little: a surface, not a sieve
        used += 1

    if d_xyz:
        xyz = np.concatenate(d_xyz)
        rgb = np.concatenate(d_rgb)
        size = np.concatenate(d_size)
        note = f"dense, {used} keyframe(s) of the depth pass"
    else:
        xyz, rgb, size = pts, sparse_rgb, dist * 6.0 / k[0]
        note = "feature points of the solve"
    if len(xyz) > max_points:
        pick = np.sort(rng.choice(len(xyz), max_points, replace=False))
        xyz, rgb, size = xyz[pick], rgb[pick], size[pick]
    return DisplayCloud(
        xyz=np.asarray(xyz, np.float32),
        rgb=_lift(np.asarray(rgb, np.float32)),
        size=np.asarray(size, np.float32),
        dense=bool(d_xyz),
        note=f"{len(xyz)} points: {note}",
    )


def write_cloud_obj(cloud: DisplayCloud, path: Path, face: np.ndarray) -> tuple[Path, Path]:
    """`path` (.obj, one quad per point, turned towards `face`: the middle of
    the camera path) and the colour atlas `path.with_suffix(".png")`."""
    import OpenImageIO as oiio

    n = len(cloud)
    side = max(1, int(np.ceil(np.sqrt(n))))
    atlas = np.zeros((side * side, 3), np.float32)
    atlas[:n] = cloud.rgb
    png = path.with_suffix(".png")
    spec = oiio.ImageSpec(side, side, 3, "uint8")
    spec.attribute("oiio:ColorSpace", "sRGB")
    out = oiio.ImageOutput.create(str(png))
    if out is None or not out.open(str(png), spec):
        raise RuntimeError(f"display cloud: cannot write {png}: {oiio.geterror()}")
    out.write_image((atlas.reshape(side, side, 3) * 255.0 + 0.5).astype(np.uint8))
    out.close()

    xyz = cloud.xyz.astype(np.float64)
    fwd = xyz - np.asarray(face, np.float64)
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-9)
    up = np.cross(right, fwd)
    half = (cloud.size.astype(np.float64) / 2.0)[:, None]
    quads = np.stack(
        [
            xyz - right * half - up * half,
            xyz + right * half - up * half,
            xyz + right * half + up * half,
            xyz - right * half + up * half,
        ],
        1,
    )
    idx = np.arange(n)
    uv = np.stack([(idx % side + 0.5) / side, 1.0 - (idx // side + 0.5) / side], 1)
    a, t = idx * 4 + 1, idx + 1
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(
            "# LiveActionAOV display point cloud (Y-up, same world as the camera).\n"
            f"# {cloud.note}. One small quad per point; colours are in {png.name}.\n"
        )
        np.savetxt(fh, quads.reshape(-1, 3), fmt="v %.5f %.5f %.5f")
        np.savetxt(fh, uv, fmt="vt %.6f %.6f")
        np.savetxt(
            fh, np.stack([a, t, a + 1, t, a + 2, t, a + 3, t], 1), fmt="f %d/%d %d/%d %d/%d %d/%d"
        )
    return path, png


__all__ = ["DisplayCloud", "PassReader", "PlateReader", "build_display_cloud", "write_cloud_obj"]
