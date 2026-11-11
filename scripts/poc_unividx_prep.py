"""POC harness: prep a plate for UniVidX intrinsic (albedo) inference.

Dev tool, not shipped — lives in `scripts/` and is run by hand to
validate a new vendored model BEFORE we commit to vendoring it. This one
de-risks UniVidX: it turns a LiveActionAOV plate sequence into the
display-space PNG frames UniVidX consumes, using the *same* reader +
display transform the real passes use, so what you feed the model is what
the model would see in production.

What this script DOES (fully implemented, verified against our own code):
  1. Reads an EXR plate sequence via OIIOExrReader (+ optional proxy).
  2. Applies the clip-uniform display transform (auto-exposure → AgX →
     sRGB), exactly like the depth/normals passes.
  3. Writes numbered frames into <out>/<video_name>/00000.rgb.jpg ...
     — the folder layout UniVidX's inference script expects.
  4. Probes + prints total VRAM so you know whether the 14B model fits.

What this script does NOT do (on purpose):
  - It does NOT run UniVidX. The 14B model isn't vendored yet — that's
    the whole point of validating first. Instead it prints the upstream
    steps (config-driven: mode R2AIN in a YAML, not CLI flags) to run
    against the prepped frames. VERIFY every path/key against the current
    upstream README before trusting it — the interface drifts:
    https://github.com/houyuanchen111/UniVidX

After you run upstream inference and eyeball the albedo quality +
measure runtime, report back and we'll vendor the Apache-2.0 inference
under src/live_action_aov/vendored/univid_x/ and wire
passes/intrinsic/univid_x.py. See docs/albedo-unividx.md.

Usage:
    uv run python scripts/poc_unividx_prep.py \
        --folder "Y:/path/to/plate/v001" \
        --pattern "shot.####.exr" \
        --first 1009 --last 1031 \
        --out ./poc_unividx_frames \
        --proxy 960 \
        --colorspace lin_rec709
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _probe_vram() -> None:
    """Print total VRAM of GPU 0 and the UniVidX fit verdict. Best-effort."""
    try:
        import torch
    except ImportError:
        print("[vram] torch not importable — skipping VRAM probe.")
        return
    if not torch.cuda.is_available():
        print("[vram] CUDA not available — UniVidX (14B) needs a GPU.")
        return
    try:
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[vram] GPU 0: {name}, {total_gb:.1f} GB total")
        # 24 GB FP8 minimum per the UniVidX_ComfyUI maintainer (peak
        # 18-20 GB FP8, 32-34 GB BF16). Earlier ~15 GB Q8 reports were
        # optimistic — 24 is the honest floor.
        if total_gb >= 32:
            print("[vram] OK -- comfortable for UniVidX (FP8 or BF16).")
        elif total_gb >= 24:
            print("[vram] OK -- fits UniVidX at FP8 (~18-20 GB peak).")
        else:
            print("[vram] TOO SMALL -- UniVidX 14B needs 24 GB FP8; expect OOM.")
    except Exception as exc:
        print(f"[vram] probe failed ({type(exc).__name__}: {exc})")


def _prep_frames(args: argparse.Namespace) -> Path:
    """Extract display-space frames into UniVidX's expected layout."""
    import cv2
    import numpy as np  # noqa: F401 - used implicitly via reader arrays

    from live_action_aov.core.pass_base import DisplayTransformParams
    from live_action_aov.io.readers.display_transform_reader import (
        DisplayTransformedReader,
    )
    from live_action_aov.io.readers.oiio_exr import OIIOExrReader
    from live_action_aov.io.readers.proxy import wrap_if_proxy

    base = OIIOExrReader(Path(args.folder), args.pattern)
    base = wrap_if_proxy(base, args.proxy)

    params = DisplayTransformParams(
        input_colorspace=args.colorspace,
        tonemap="agx",
        output_eotf="srgb",
        clamp=True,
    )
    reader = DisplayTransformedReader(
        base,
        params=params,
        colorspace_override=(
            args.colorspace if args.colorspace and args.colorspace != "auto" else None
        ),
    )
    first, last = args.first, args.last
    reader.analyze((first, last))

    video_dir = Path(args.out) / args.video_name
    video_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for i, f in enumerate(range(first, last + 1)):
        rgb, _attrs = reader.read_frame(f)  # (H, W, 3) float32 [0,1], sRGB-display
        # read_frame already strips alpha → RGB (since the #32 fix), but be
        # defensive in case this script runs against an older checkout.
        if rgb.ndim == 3 and rgb.shape[-1] > 3:
            rgb = rgb[..., :3]
        u8 = (rgb.clip(0.0, 1.0) * 255.0 + 0.5).astype("uint8")
        # cv2 writes BGR; convert from our RGB.
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        out_path = video_dir / f"{i:05d}.rgb.jpg"
        cv2.imwrite(str(out_path), bgr)
        n += 1
    print(f"[prep] wrote {n} frames -> {video_dir}")
    return video_dir


def _print_upstream_commands(video_dir: Path, n_frames: int) -> None:
    print("\n" + "=" * 70)
    print(" NEXT: run UniVidX upstream inference on the prepped frames.")
    print(" UniVidX is config-driven (a YAML), NOT CLI flags. VERIFY every")
    print(" path/key against the current upstream README before trusting it")
    print(" — the interface drifts release to release.")
    print("   https://github.com/houyuanchen111/UniVidX")
    print("=" * 70)
    print(
        "\n# 1. Clone + env (one-time):\n"
        "git clone https://github.com/houyuanchen111/UniVidX.git\n"
        "cd UniVidX\n"
        "conda create -n unividx python=3.10 -y && conda activate unividx\n"
        "pip install -r requirement.txt   # NB: singular, and add: regex av\n"
    )
    print(
        "# 2. Download weights (~85 GB total — backbone + LoRA checkpoint):\n"
        'pip install "huggingface_hub[cli]"\n'
        "huggingface-cli download Wan-AI/Wan2.1-T2V-14B \\\n"
        "    --local-dir ./models/Wan-AI/Wan2.1-T2V-14B   # ~69 GB backbone\n"
        "huggingface-cli download houyuanchen/UniVidX \\\n"
        "    --local-dir ./checkpoints                     # univid_intrinsic.safetensors\n"
    )
    print(
        "# 3. Edit configs/univid_intrinsic_inference.yaml:\n"
        "#      mode: R2AIN          # RGB -> albedo + irradiance + normal\n"
        f"#      inference_rgb_path: {video_dir.resolve()}\n"
        "#      inference_albedo_path: null      # leave null for R2AIN\n"
        "#      inference_irradiance_path: null  #   (these are the targets)\n"
        "#      inference_normal_path: null\n"
        "#      experiment_name: poc_albedo\n"
        f"#    NOTE: {n_frames} frames prepped. inference_rgb_path takes a single\n"
        "#    MP4 (torchvision.io.read_video), NOT a folder. Mux the frames:\n"
        "#      ffmpeg -framerate 24 -i %05d.rgb.jpg -pix_fmt yuv420p poc.mp4\n"
        "#    The model forces 640x480 / 21 frames / 50 steps (non-4:3 input is\n"
        "#    squished). Verified: ~21 GB peak / ~33s per frame on a 5090 with\n"
        "#    CPU offload; needs a 24 GB card.\n"
    )
    print(
        "# 4. Run inference:\n"
        "python scripts/inference_univid_intrinsic.py \\\n"
        "    --config configs/univid_intrinsic_inference.yaml\n"
    )
    print(
        "# 5. Time it + watch VRAM in another shell:\n"
        "#    nvidia-smi --query-gpu=memory.used --format=csv -l 1\n"
        "# Report back: wall-clock for the clip, peak VRAM, and whether the\n"
        "# albedo looks lighting-free (no baked shadows in the base colour).\n"
    )


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp1252; force UTF-8 so non-ASCII in the
    # -h help text or status lines never raises UnicodeEncodeError (it just
    # crashed a real run on a cp1252 console). Guarded — a redirected
    # stream may not support reconfigure.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder", required=True, help="Plate folder (contains the EXR sequence)")
    p.add_argument("--pattern", required=True, help="Sequence pattern, e.g. shot.####.exr")
    p.add_argument("--first", type=int, required=True, help="First frame (inclusive)")
    p.add_argument("--last", type=int, required=True, help="Last frame (inclusive)")
    p.add_argument("--out", default="./poc_unividx_frames", help="Output root for prepped frames")
    p.add_argument("--video-name", dest="video_name", default="video_1", help="Subfolder name")
    p.add_argument(
        "--proxy", type=int, default=None, help="Proxy long-edge px (e.g. 960); omit for full res"
    )
    p.add_argument(
        "--colorspace", default="auto", help="Plate colorspace, e.g. lin_rec709 / acescg / auto"
    )
    args = p.parse_args(argv)

    _probe_vram()
    try:
        video_dir = _prep_frames(args)
    except Exception as exc:
        print(f"[error] frame prep failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    n_frames = args.last - args.first + 1
    _print_upstream_commands(video_dir, n_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
