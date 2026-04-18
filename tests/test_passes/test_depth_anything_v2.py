"""DepthAnythingV2Pass — contract tests with a bypassed model loader.

We short-circuit `_load_model` so tests don't hit HuggingFace. A tiny fake
model returns a spatially-varying depth field we can reason about. The real
model download + inference is exercised by the marked-slow integration
test, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from live_action_aov.io.channels import CH_Z, CH_Z_RAW  # noqa: E402
from live_action_aov.passes.depth.depth_anything_v2 import DepthAnythingV2Pass  # noqa: E402


class _FakePass(DepthAnythingV2Pass):
    """Subclass that uses a deterministic fake instead of a real HF model.

    The fake's depth equals `mean(image) * base + frame_seed` so per-frame
    raw values differ — which lets us verify per-clip normalization actually
    spans [0, 1] across frames rather than being computed per-frame.
    """

    def __init__(self, params=None, *, frame_seed: float = 0.0) -> None:
        super().__init__(params)
        self._frame_seed = float(frame_seed)

    def _load_model(self) -> None:  # type: ignore[override]
        # Idempotent no-op — no HF download.
        if self._model is not None:
            return
        self._model = object()
        self._processor = None
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def preprocess(self, frames: np.ndarray):  # type: ignore[override]
        if frames.ndim != 4 or frames.shape[0] != 1 or frames.shape[-1] != 3:
            raise ValueError(f"Bad input shape {frames.shape}")
        self._load_model()
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        # Pretend we resized to 56x56 — multiples of 14, the ViT patch size.
        inf_h, inf_w = 56, 56
        img = np.clip(frames[0], 0.0, 1.0).astype(np.float32)
        img_mean = float(img.mean())
        # Emit a horizontal gradient scaled by image mean + frame seed.
        grad = np.linspace(0.0, 1.0, inf_w, dtype=np.float32)[None, :].repeat(inf_h, 0)
        depth = grad * (0.5 + img_mean) + self._frame_seed
        t = torch.from_numpy(depth)[None, None]  # (1, 1, h, w) but we want (1, h, w)
        return {
            "pixel_values": t.squeeze(1),
            "plate_shape": (plate_h, plate_w),
            "_depth_stub": t,
        }

    def infer(self, tensor):  # type: ignore[override]
        # Reuse the pre-synthesized depth from preprocess so the fake is
        # deterministic regardless of model state.
        return {
            "depth": tensor["_depth_stub"].squeeze(1),  # (1, h, w)
            "plate_shape": tensor["plate_shape"],
        }


class _FakeReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._first = 1

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._frames[idx - self._first], {}


def _make_frame(h: int = 32, w: int = 48, shade: float = 0.5) -> np.ndarray:
    return np.full((h, w, 3), shade, dtype=np.float32)


def test_postprocess_emits_z_and_zraw_at_plate_resolution() -> None:
    pass_ = _FakePass()
    pair = _make_frame()[None, ...]  # (1, H, W, 3)
    model_in = pass_.preprocess(pair)
    model_out = pass_.infer(model_in)
    channels = pass_.postprocess(model_out)

    assert CH_Z in channels and CH_Z_RAW in channels
    assert channels[CH_Z].shape == (32, 48)
    assert channels[CH_Z].dtype == np.float32
    # Single-frame postprocess doesn't normalize — Z == Z_raw.
    assert np.array_equal(channels[CH_Z], channels[CH_Z_RAW])


def test_run_shot_per_clip_normalization_spans_zero_to_one() -> None:
    """Three frames with different raw-depth offsets — after run_shot the
    combined Z span across ALL frames should hit both 0 and 1.
    """
    frames = [_make_frame(shade=0.5) for _ in range(3)]
    # Vary the frame seeds to force distinct raw ranges per frame.
    pass_ = _FakePass(frame_seed=0.0)
    reader = _FakeReader(frames)

    # Hack: swap pass's seed between frames by reading frames with different
    # shades. We can drive this via the shade field instead — deeper shade
    # means larger mean and larger depth output.
    frames[0][...] = 0.2   # darker → lower offset
    frames[2][...] = 0.9   # brighter → higher offset
    out = pass_.run_shot(reader, frame_range=(1, 3))

    assert sorted(out) == [1, 2, 3]
    for f in (1, 2, 3):
        assert CH_Z in out[f] and CH_Z_RAW in out[f]
        assert out[f][CH_Z].shape == (32, 48)
        assert out[f][CH_Z_RAW].shape == (32, 48)

    # Combined min/max across all frames must span [0, 1]. Per-clip norm
    # means exactly one pixel hits 0 (the globally brightest raw, flipped)
    # and one hits 1 (the globally dimmest raw, flipped).
    stacked = np.stack([out[f][CH_Z] for f in (1, 2, 3)])
    assert stacked.min() == pytest.approx(0.0, abs=1e-5)
    assert stacked.max() == pytest.approx(1.0, abs=1e-5)

    # Raw values are monotonically varying per-frame (darker shade → lower).
    raw_means = [float(out[f][CH_Z_RAW].mean()) for f in (1, 2, 3)]
    assert raw_means[0] < raw_means[1] < raw_means[2]


def test_emit_artifacts_exposes_normalization_constants() -> None:
    pass_ = _FakePass()
    reader = _FakeReader([_make_frame(shade=0.3), _make_frame(shade=0.7)])
    pass_.run_shot(reader, frame_range=(1, 2))
    artifacts = pass_.emit_artifacts()
    assert "depth_norm_min" in artifacts and "depth_norm_max" in artifacts
    min_arr = next(iter(artifacts["depth_norm_min"].values()))
    max_arr = next(iter(artifacts["depth_norm_max"].values()))
    assert min_arr.shape == (1,) and max_arr.shape == (1,)
    assert min_arr[0] < max_arr[0]


def test_license_is_commercial_safe() -> None:
    lic = DepthAnythingV2Pass.declared_license()
    assert lic.spdx == "Apache-2.0"
    assert lic.commercial_use is True


def test_refuses_noncommercial_variants() -> None:
    with pytest.raises(ValueError, match="CC-BY-NC"):
        DepthAnythingV2Pass({"variant": "large"})
    with pytest.raises(ValueError, match="CC-BY-NC"):
        DepthAnythingV2Pass({"variant": "giant"})


def test_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unknown variant"):
        DepthAnythingV2Pass({"variant": "nano"})
