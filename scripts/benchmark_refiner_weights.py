"""Benchmark soft-matte refiner weights on a real plate — edge quality A/B.

Answers "which BiRefNet-family weights give the best soft edges for our
matte, and is a paid model (RMBG-2.0) worth it?" by running each candidate
through the SHIPPING inference path (`BiRefNetRefinerPass._birefnet_alpha`)
on one real frame and reporting the soft-edge fraction + writing preview
PNGs so a human can judge hair by eye.

Why in-repo (not a throwaway): the model choice drives delivery quality on
paid work, so the comparison must be reproducible and auditable, not a
scratchpad hack. Re-run it whenever a new candidate model appears.

All candidates share the BiRefNet architecture + HF loader, so switching is
a single `model_id`. Licences differ and MATTER for commercial delivery:
- general  ZhengPeng7/BiRefNet          MIT weights, DIS5K data — cleanest, harder edges
- matting  ZhengPeng7/BiRefNet-matting  MIT weights, but NC training data (P3M, Distinctions-646...)
- portrait ZhengPeng7/BiRefNet-portrait MIT weights, NC training data (P3M) — current default
- rmbg2    briaai/RMBG-2.0              CC BY-NC; COMMERCIAL USE NEEDS A PAID BRIA LICENCE; gated on HF

`rmbg2` is gated — accept the licence at https://huggingface.co/briaai/RMBG-2.0
with your HF account first, or this script skips it with a clear message.

**Side effects**: downloads each model into the HF cache on first use; runs
on CUDA if available. Not a unit test — it needs weights + a GPU.

Run::

    uv run python scripts/benchmark_refiner_weights.py \
        --plate  PATH/to/plate.####.exr \
        --sidecar PATH/to/utility.####.exr \
        --mask-channel mask.person \
        --models general,matting,portrait,rmbg2 \
        --out scripts/_refiner_bench
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MODELS: dict[str, str] = {
    "general": "ZhengPeng7/BiRefNet",
    "matting": "ZhengPeng7/BiRefNet-matting",
    "portrait": "ZhengPeng7/BiRefNet-portrait",
    "rmbg2": "briaai/RMBG-2.0",
}


def _read_exr(path: str, channel: str | None = None):
    import OpenImageIO as oiio

    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise SystemExit(f"cannot open {path}: {oiio.geterror()}")
    spec = inp.spec()
    px = np.asarray(inp.read_image(format=oiio.FLOAT)).reshape(
        spec.height, spec.width, spec.nchannels
    )
    names = list(spec.channelnames)
    inp.close()
    if channel is None:
        return px, names
    if channel not in names:
        raise SystemExit(f"channel {channel!r} not in {path} (have: {names})")
    return px[..., names.index(channel)], names


def _srgb(rgb_lin: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb_lin.astype(np.float32), 0, None)
    s = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return np.clip(s, 0, 1).astype(np.float32)


def _kernel(n: int) -> np.ndarray:
    return np.ones((2 * n + 1, 2 * n + 1), np.uint8)


def _is_gated_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__}: {exc}".lower()
    return any(k in s for k in ("gated", "401", "restricted", "awaiting", "access to model", "403"))


def _run_vitmatte(crop: np.ndarray, hard_crop: np.ndarray) -> np.ndarray:
    """ViTMatte through the shipping pass: trimap from the hard mask,
    single-frame stacks, full _refine_instance path."""
    from live_action_aov.passes.matte.vitmatte import ViTMatteRefinerPass

    p = ViTMatteRefinerPass({})
    p._model = None
    alpha = p._refine_instance(crop[None, ...], hard_crop[None, ...].astype(np.float32))[0]
    p._model = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return alpha


def _run_model(model_id: str, crop: np.ndarray, infer_size: int) -> np.ndarray:
    """Refine one crop through the shipping inference path."""
    from live_action_aov.passes.matte.birefnet import BiRefNetRefinerPass

    p = BiRefNetRefinerPass(
        {"model_id": model_id, "infer_size": infer_size, "precision": "fp16",
         "hard_mask_dilate": 5, "crop_pad_fraction": 0.0}
    )
    p._model = None
    p._load_model()
    alpha = np.clip(p._birefnet_alpha(crop).astype(np.float32), 0, 1)
    p._model = None
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return alpha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plate", required=True, help="source plate EXR (linear)")
    ap.add_argument("--sidecar", required=True, help="utility EXR holding the hard mask")
    ap.add_argument("--mask-channel", default="mask.person", help="hard-mask channel to refine")
    ap.add_argument("--models", default="general,matting,portrait,rmbg2")
    ap.add_argument("--infer-size", type=int, default=1024)
    ap.add_argument("--out", default="scripts/_refiner_bench")
    args = ap.parse_args()

    import cv2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plate, _ = _read_exr(args.plate)
    srgb = _srgb(plate[..., :3])
    H, W = srgb.shape[:2]

    hard, _ = _read_exr(args.sidecar, args.mask_channel)
    if hard.shape != (H, W):
        hard = cv2.resize(hard, (W, H), interpolation=cv2.INTER_NEAREST)
    hard = (hard > 0.5).astype(np.uint8)
    if hard.sum() == 0:
        raise SystemExit(f"{args.mask_channel} is empty on this frame — pick another frame/channel")

    ys, xs = np.nonzero(hard)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    ph, pw = int((y1 - y0) * 0.35), int((x1 - x0) * 0.35)
    y0, y1 = max(0, y0 - ph), min(H, y1 + ph)
    x0, x1 = max(0, x0 - pw), min(W, x1 + pw)
    crop = srgb[y0:y1, x0:x1]
    hard_crop = hard[y0:y1, x0:x1]
    band = (cv2.dilate(hard_crop, _kernel(30)) & (1 - cv2.erode(hard_crop, _kernel(4)))).astype(bool)

    def soft_pct(a: np.ndarray, region: np.ndarray | None = None) -> float:
        m = (a > 0.02) & (a < 0.98)
        if region is not None:
            return 100.0 * (m & region).sum() / max(region.sum(), 1)
        return 100.0 * m.mean()

    cv2.imwrite(str(out_dir / "crop_plate.png"),
                cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    print(f"plate {W}x{H}  crop {crop.shape[1]}x{crop.shape[0]}  "
          f"mask={args.mask_channel} ({100 * hard.mean():.1f}% cover)\n")
    print(f"{'model':<12}{'model_id':<30}{'soft% whole':>13}{'soft% hair':>12}")
    print("-" * 67)

    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        model_id = MODELS.get(key, key)
        try:
            if key == "vitmatte":
                # Different arch: trimap-guided, native HF (MIT code+weights).
                model_id = "hustvl/vitmatte-base-composition-1k"
                a = _run_vitmatte(crop, hard_crop)
            else:
                a = _run_model(model_id, crop, args.infer_size)
        except Exception as exc:
            if _is_gated_error(exc):
                print(f"{key:<12}{model_id:<30}  GATED - accept the licence at "
                      f"https://huggingface.co/{model_id}")
            else:
                print(f"{key:<12}{model_id:<30}  FAILED: {type(exc).__name__}: {exc}")
            continue
        print(f"{key:<12}{model_id:<30}{soft_pct(a):>12.2f}%{soft_pct(a, band):>11.2f}%")
        cv2.imwrite(str(out_dir / f"alpha_{key}.png"), (a * 255).astype(np.uint8))

    print(f"\nPNGs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
