# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Gaussian splat of the static scene, trained on the shot's own camera solve.

Consumes the `camera_solve` artifact of `camera_pycolmap` (poses, plate-res
intrinsics, sparse points as initialisation) and trains a 3D Gaussian
splat with gsplat (Apache-2.0, nerfstudio) on the plate itself. Then it
re-renders every solved frame through the solved camera:

- `splat.R/G/B` — the static scene, scene-linear in the plate's colourspace.
  When a SAM 3 matte ran in the same job, every tracked object is excluded
  from the training loss, so this is a **clean plate**: what was behind the
  person, recovered from the frames where the camera saw past them.
- `splat.A` — splat coverage (0 = nothing reconstructed there).
- `splat.Z` — camera-space depth of the splat (camera-track units, same
  frame as the Nuke camera): multi-view consistent by construction.

Sidecar: `splat/<shot>.splat.ply` — standard 3DGS PLY in the SAME
Y-up world as `<shot>.camera.nk`, so it lines up with the exported camera
in any splat viewer / DCC that reads 3DGS PLY.

Honest limits:
- It only knows what the plate saw. Areas never revealed (behind a person
  who never moves out of the way, outside the camera move) come back empty
  (`splat.A` ~ 0) or smeared — check `splat.A` before trusting a patch.
- Needs a good camera solve (it inherits every error of the track).
- Training is per-shot, minutes not seconds (7k steps by default).
- Colour: trained on a perceptual encoding of scene-linear (sRGB-curve,
  extended above 1 so highlights survive), decoded back to linear.

gsplat must be COMPILED for the local GPU (no prebuilt Windows wheel for
torch 2.11 / CUDA 12.8). See docs/gaussian_splat.md and
scripts/build_gsplat_wheel.bat.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.core.output_layout import SPLAT_DIR, run_stem, sub_dir
from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    SidecarSpec,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import CH_SPLAT_A, CH_SPLAT_B, CH_SPLAT_G, CH_SPLAT_R, CH_SPLAT_Z

_log = logging.getLogger(__name__)

_F = np.diag([1.0, -1.0, -1.0])  # OpenGL <-> OpenCV camera-local flip
_SH_C0 = 0.28209479177387814

_GSPLAT_HELP = (
    "gsplat is not installed (or not built for this GPU). It needs a one-time "
    "local build: run scripts\\build_gsplat_wheel.bat (CUDA Toolkit 12.8 + "
    "Visual Studio 2022 Build Tools required). See docs/gaussian_splat.md."
)


class GsplatSplatPass(UtilityPass):
    name = "splat_gsplat"
    version = "0.1.0"
    license = License(
        spdx="Apache-2.0",
        commercial_use=True,
        commercial_tool_resale=True,
        notes="gsplat (nerfstudio-project) is Apache-2.0; no pretrained weights.",
    )
    pass_type = PassType.SCENE_3D
    temporal_mode = TemporalMode.VIDEO_CLIP
    # Train and output in scene-linear: the clean plate must comp straight
    # back over the plate.
    input_colorspace = "scene_linear"

    produces_channels = [
        ChannelSpec(
            name=CH_SPLAT_R, description="Splat re-render R (scene-linear, plate colourspace)"
        ),
        ChannelSpec(name=CH_SPLAT_G, description="Splat re-render G"),
        ChannelSpec(name=CH_SPLAT_B, description="Splat re-render B"),
        ChannelSpec(name=CH_SPLAT_A, description="Splat coverage [0,1]"),
        ChannelSpec(
            name=CH_SPLAT_Z, description="Splat camera-space depth (track units, 0 = empty)"
        ),
    ]
    produces_sidecars = [SidecarSpec(name="splat", format="ply")]
    requires_artifacts = ["camera_solve"]
    optional_artifacts = ["sam3_hard_masks"]

    DEFAULT_PARAMS: dict[str, Any] = {
        "iterations": 7000,
        # Long edge of the training images; renders are always at plate res.
        "train_long_edge": 2048,
        "sh_degree": 3,
        "ssim_lambda": 0.2,
        # Every Nth solved frame is held out of training to measure quality
        # honestly (PSNR on frames the splat never saw). 0 = train on all.
        "holdout_every": 8,
        # SAM 3 exclusion: [] = every tracked object; or a list of labels.
        "mask_objects": True,
        "mask_labels": [],
        "mask_dilate_frac": 0.01,
        "seed": 0,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._solve: Any = None
        self._hard_masks: dict[int, dict[str, Any]] = {}
        self._splats: dict[str, np.ndarray] | None = None
        self.report: dict[str, Any] = {}

    # --- unused per-frame hooks ---
    def preprocess(self, frames: np.ndarray) -> Any:  # pragma: no cover
        raise NotImplementedError("splat_gsplat is VIDEO_CLIP; use run_shot().")

    def infer(self, tensor: Any) -> Any:  # pragma: no cover
        raise NotImplementedError("splat_gsplat is VIDEO_CLIP; use run_shot().")

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:  # pragma: no cover
        raise NotImplementedError("splat_gsplat is VIDEO_CLIP; use run_shot().")

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        cam = artifacts.get("camera_solve") or {}
        self._solve = next(iter(cam.values())) if cam else None
        hard = artifacts.get("sam3_hard_masks") or {}
        self._hard_masks = next(iter(hard.values())) if hard else {}

    # --- main ---
    def run_shot(
        self, reader: Any, frame_range: tuple[int, int]
    ) -> dict[int, dict[str, np.ndarray]]:
        try:
            import torch
            from gsplat import rasterization
            from gsplat.strategy import DefaultStrategy
        except ImportError as e:
            raise ImportError(_GSPLAT_HELP) from e
        if not torch.cuda.is_available():
            raise RuntimeError("splat_gsplat needs a CUDA GPU.")
        if self._solve is None or not self._solve.extrinsics:
            raise RuntimeError(
                "splat_gsplat needs a camera solve: enable 'Camera track' in the same job "
                "(the solve is passed in memory, not read from disk)."
            )
        solve = self._solve
        dev = torch.device("cuda")
        torch.manual_seed(int(self.params["seed"]))
        np.random.seed(int(self.params["seed"]))

        frames = [e.frame for e in solve.extrinsics if frame_range[0] <= e.frame <= frame_range[1]]
        if len(frames) < 3:
            raise RuntimeError(f"splat_gsplat: only {len(frames)} solved frames in range.")
        intr = solve.intrinsics
        pw, ph = int(intr.width_px), int(intr.height_px)
        s = min(1.0, float(self.params["train_long_edge"]) / max(pw, ph))
        tw, th = max(16, round(pw * s)), max(16, round(ph * s))

        # Plate (scene-linear) -> perceptual encoding, at train res.
        import cv2

        imgs, masks = [], []
        mask_stack = _object_masks(self._hard_masks, self.params)
        for f in frames:
            arr, _ = reader.read_frame(f)
            arr = np.nan_to_num(
                np.asarray(arr, np.float32)[..., :3], nan=0.0, posinf=0.0, neginf=0.0
            )
            if arr.shape[:2] != (th, tw):
                arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_AREA)
            imgs.append(encode(arr))
            masks.append(_frame_mask(mask_stack, f, (th, tw), self.params))
        gt = torch.from_numpy(np.stack(imgs)).to(dev)  # [N,H,W,3]
        keep = torch.from_numpy(np.stack(masks)).to(dev)  # [N,H,W] 1 = train on it
        masked_frac = float(1.0 - keep.mean().item())

        viewmats = (
            torch.from_numpy(np.stack([_viewmat(e) for e in solve.extrinsics if e.frame in frames]))
            .float()
            .to(dev)
        )
        K_plate = np.array(
            [[intr.fx_px, 0, intr.cx_px], [0, intr.fy_px, intr.cy_px], [0, 0, 1]], np.float32
        )
        K_train = K_plate.copy()
        K_train[0] *= tw / pw
        K_train[1] *= th / ph
        Kt = torch.from_numpy(K_train).to(dev)

        every = int(self.params["holdout_every"])
        holdout = [i for i in range(len(frames)) if every > 0 and i % every == every // 2]
        train_idx = [i for i in range(len(frames)) if i not in holdout]

        # --- init from the sparse cloud (same GL world as the camera) ---
        pts = np.array([p.xyz for p in solve.points], np.float32)
        if len(pts) < 100:
            raise RuntimeError(f"splat_gsplat: sparse cloud too small ({len(pts)} points).")
        cam_pos = np.array([e.translation for e in solve.extrinsics], np.float32)
        # Forward-facing plates: the camera barely moves compared with the
        # scene depth, so normalise the densify thresholds by scene distance,
        # not by camera extent (which would prune every background splat).
        scene_scale = float(np.median(np.linalg.norm(pts - cam_pos.mean(0), axis=1))) * 1.1
        means = torch.from_numpy(pts).to(dev)
        d2 = _knn_mean_sq_dist(means, k=3)
        sh_deg = int(self.params["sh_degree"])
        n = means.shape[0]
        rgb0 = _init_colors(pts, solve.extrinsics, frames, imgs, K_train, tw, th)
        colors = torch.zeros((n, (sh_deg + 1) ** 2, 3), device=dev)
        colors[:, 0, :] = (torch.from_numpy(rgb0).to(dev) - 0.5) / _SH_C0
        params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means),
                "scales": torch.nn.Parameter(torch.log(torch.sqrt(d2)).unsqueeze(-1).repeat(1, 3)),
                "quats": torch.nn.Parameter(
                    torch.nn.functional.normalize(torch.rand((n, 4), device=dev), dim=-1)
                ),
                "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=dev))),
                "sh0": torch.nn.Parameter(colors[:, :1, :].contiguous()),
                "shN": torch.nn.Parameter(colors[:, 1:, :].contiguous()),
            }
        )
        lrs = {
            "means": 1.6e-4 * scene_scale,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 2.5e-3 / 20,
        }
        optimizers = {
            k: torch.optim.Adam([{"params": params[k], "lr": lr, "name": k}], eps=1e-15)
            for k, lr in lrs.items()
        }
        iters = int(self.params["iterations"])
        sched = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max(iters, 1))
        )
        strategy = DefaultStrategy(
            verbose=False, refine_stop_iter=int(iters * 0.6), reset_every=3000
        )
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=scene_scale)
        lam = float(self.params["ssim_lambda"])

        t0 = time.time()
        order = np.random.permutation(np.repeat(train_idx, math.ceil(iters / len(train_idx))))[
            :iters
        ]
        for step in range(iters):
            i = int(order[step])
            sh_use = min(step // 1000, sh_deg)
            img, _alpha, info = rasterization(
                means=params["means"],
                quats=params["quats"],
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=torch.cat([params["sh0"], params["shN"]], 1),
                viewmats=viewmats[i : i + 1],
                Ks=Kt[None],
                width=tw,
                height=th,
                sh_degree=sh_use,
                packed=False,
            )
            strategy.step_pre_backward(params, optimizers, state, step, info)
            pred = img[0]
            w = keep[i][..., None]
            l1 = (torch.abs(pred - gt[i]) * w).sum() / (w.sum() * 3 + 1e-6)
            ssim_map = _ssim_map(pred, gt[i])
            dssim = 1.0 - (ssim_map * keep[i]).sum() / (keep[i].sum() + 1e-6)
            loss = (1 - lam) * l1 + lam * dssim
            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            sched.step()
            strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
            if step % 1000 == 0:
                _log.info(
                    "splat step %d/%d loss %.4f gaussians %d",
                    step,
                    iters,
                    loss.item(),
                    params["means"].shape[0],
                )
        train_s = time.time() - t0

        # --- quality on held-out frames (never trained on) ---
        with torch.no_grad():

            def render(i: int, K: Any, w_: int, h_: int) -> tuple[Any, Any]:
                out, alpha, _ = rasterization(
                    means=params["means"],
                    quats=params["quats"],
                    scales=torch.exp(params["scales"]),
                    opacities=torch.sigmoid(params["opacities"]),
                    colors=torch.cat([params["sh0"], params["shN"]], 1),
                    viewmats=viewmats[i : i + 1],
                    Ks=K[None],
                    width=w_,
                    height=h_,
                    sh_degree=sh_deg,
                    packed=False,
                    render_mode="RGB+ED",
                )
                return out[0], alpha[0, ..., 0]

            psnr = []
            for i in holdout:
                pred, _a = render(i, Kt, tw, th)
                m = keep[i] > 0
                mse = ((pred[..., :3].clamp(0, 1) - gt[i].clamp(0, 1)) ** 2)[m].mean()
                psnr.append(float(-10 * torch.log10(mse + 1e-10)))
            self.report = {
                "gaussians": int(params["means"].shape[0]),
                "train_seconds": round(train_s, 1),
                "train_res": [tw, th],
                "holdout_frames": [frames[i] for i in holdout],
                "holdout_psnr_db": round(float(np.mean(psnr)), 2) if psnr else None,
                "masked_fraction": round(masked_frac, 4),
                "scene_scale": scene_scale,
            }
            _log.info(
                "Splat: %d gaussians, %.0f s training at %dx%d, held-out PSNR %s dB, %.1f%% of pixels masked",
                self.report["gaussians"],
                train_s,
                tw,
                th,
                self.report["holdout_psnr_db"],
                100 * masked_frac,
            )

            # --- re-render every solved frame at plate resolution ---
            Kp = torch.from_numpy(K_plate).to(dev)
            out: dict[int, dict[str, np.ndarray]] = {}
            for i, f in enumerate(frames):
                rgbd, alpha = render(i, Kp, pw, ph)
                rgb = decode(rgbd[..., :3].clamp_min(0).cpu().numpy())
                a = alpha.cpu().numpy().astype(np.float32)
                # "RGB+ED" depth is alpha-weighted: expected depth = D / alpha.
                z = (rgbd[..., 3] / alpha.clamp_min(1e-4)).cpu().numpy().astype(np.float32)
                z[a < 1e-3] = 0.0
                out[f] = {
                    CH_SPLAT_R: rgb[..., 0],
                    CH_SPLAT_G: rgb[..., 1],
                    CH_SPLAT_B: rgb[..., 2],
                    CH_SPLAT_A: a,
                    CH_SPLAT_Z: z,
                }
            self._splats = {k: v.detach().cpu().numpy() for k, v in params.items()}
        torch.cuda.empty_cache()
        return out

    def emit_sidecars(self, shot: Any) -> dict[str, Path]:
        if self._splats is None:
            return {}
        out_dir = sub_dir(shot, SPLAT_DIR)
        stem = run_stem(shot)
        path = out_dir / f"{stem}.splat.ply"
        write_3dgs_ply(self._splats, path)
        import json

        (out_dir / f"{stem}.splat.json").write_text(
            json.dumps(self.report, indent=2), encoding="utf-8"
        )
        return {"splat": path}

    def unload(self) -> None:
        self._splats = None


# ---------------------------------------------------------------------------
# Helpers (pure numpy where possible -> unit-testable without gsplat)
# ---------------------------------------------------------------------------


def encode(lin: np.ndarray) -> np.ndarray:
    """Scene-linear -> sRGB-curve encoding, extended above 1 (invertible)."""
    x = np.maximum(lin, 0.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(
        np.float32
    )


def decode(enc: np.ndarray) -> np.ndarray:
    y = np.maximum(enc, 0.0)
    return np.where(y <= 0.04045, y / 12.92, np.power((y + 0.055) / 1.055, 2.4)).astype(np.float32)


def _viewmat(e: Any) -> np.ndarray:
    """GL world_from_camera (Nuke camera) -> OpenCV world-to-camera 4x4 (gsplat)."""
    R = np.asarray(e.rotation_matrix, np.float64)
    t = np.asarray(e.translation, np.float64)
    R_cw = _F @ R.T
    M = np.eye(4)
    M[:3, :3] = R_cw
    M[:3, 3] = -R_cw @ t
    return M.astype(np.float32)


def _object_masks(
    hard_masks: dict[int, dict[str, Any]], params: dict[str, Any]
) -> list[dict[str, Any]]:
    if not params.get("mask_objects", True) or not hard_masks:
        return []
    wanted = {str(x).lower() for x in (params.get("mask_labels") or [])}
    return [
        {"index": {fr: i for i, fr in enumerate(t.get("frames") or [])}, "stack": t["stack"]}
        for t in hard_masks.values()
        if not wanted or str(t.get("label", "")).lower() in wanted
    ]


def _frame_mask(
    tracks: list[dict[str, Any]], frame: int, hw: tuple[int, int], params: dict[str, Any]
) -> np.ndarray:
    """1 = pixel used for training, 0 = SAM 3 object (dilated)."""
    import cv2

    h, w = hw
    union = np.zeros((h, w), np.uint8)
    for t in tracks:
        i = t["index"].get(frame)
        if i is None:
            continue
        m = (np.asarray(t["stack"][i]) > 0).astype(np.uint8)
        if m.shape != (h, w):
            m = np.asarray(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST), np.uint8)
        union |= m
    excluded: np.ndarray = union
    r = round(float(params.get("mask_dilate_frac", 0.01)) * max(h, w))
    if union.any() and r > 0:
        excluded = cv2.dilate(
            union, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        )
    return (excluded == 0).astype(np.float32)


def _init_colors(
    pts: np.ndarray,
    extrinsics: list[Any],
    frames: list[int],
    imgs: list[np.ndarray],
    K: np.ndarray,
    w: int,
    h: int,
) -> np.ndarray:
    """Colour each sparse point from the first frame that sees it (the
    camera track's cloud is grey: it was solved on CLAHE greyscale)."""
    rgb = np.full((len(pts), 3), 0.5, np.float32)
    todo = np.ones(len(pts), bool)
    by_frame = {e.frame: e for e in extrinsics}
    for f, img in zip(frames, imgs, strict=True):
        if not todo.any():
            break
        V = _viewmat(by_frame[f]).astype(np.float64)
        c = pts @ V[:3, :3].T + V[:3, 3]
        z = c[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * c[:, 0] / z + K[0, 2]
            v = K[1, 1] * c[:, 1] / z + K[1, 2]
        ok = todo & (z > 1e-6) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        idx = np.nonzero(ok)[0]
        rgb[idx] = img[v[idx].astype(int), u[idx].astype(int)]
        todo[idx] = False
    return np.clip(rgb, 0.0, 1.0)


def _knn_mean_sq_dist(x: Any, k: int = 3) -> Any:
    import torch

    out = torch.empty(x.shape[0], device=x.device)
    for s in range(0, x.shape[0], 4096):
        d = torch.cdist(x[s : s + 4096], x)
        out[s : s + 4096] = (torch.topk(d, k + 1, largest=False).values[:, 1:] ** 2).mean(-1)
    return out.clamp_min(1e-7)


def _ssim_map(a: Any, b: Any) -> Any:
    """SSIM map (H, W) for HxWx3 tensors, 11x11 Gaussian window, sigma 1.5."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, device=a.device, dtype=a.dtype) - 5) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)
    x = a.permute(2, 0, 1)[None]
    y = b.permute(2, 0, 1)[None]

    def blur(t: Any) -> Any:
        return F.conv2d(t, win, padding=5, groups=3)

    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return m[0].mean(0)


def write_3dgs_ply(splats: dict[str, np.ndarray], path: Path) -> None:
    """Standard (INRIA) 3DGS binary PLY: x y z nx ny nz f_dc_* f_rest_*
    opacity scale_* rot_* — raw (pre-activation) values, as viewers expect."""
    means = splats["means"].astype(np.float32)
    n = means.shape[0]
    f_dc = splats["sh0"].reshape(n, 3).astype(np.float32)
    # gsplat shN is [N, K, 3]; INRIA stores channel-major [N, 3, K].
    f_rest = np.transpose(splats["shN"], (0, 2, 1)).reshape(n, -1).astype(np.float32)
    cols = [
        means,
        np.zeros_like(means),
        f_dc,
        f_rest,
        splats["opacities"].reshape(n, 1),
        splats["scales"],
        splats["quats"],
    ]
    data = np.concatenate([c.astype(np.float32) for c in cols], axis=1)
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(np.ascontiguousarray(data, dtype="<f4").tobytes())


__all__ = ["GsplatSplatPass", "decode", "encode", "write_3dgs_ply"]
