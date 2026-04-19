"""DepthProPass — contract tests with a bypassed model loader.

Metric depth backend. Tests assert:
  - license is non-commercial (Apple ML Research License)
  - `Z` / `Z_raw` are both metric (no per-clip flip-normalization)
  - `depth.confidence` is produced (clipped to [0, 1])
  - emit_artifacts emits `depth_metric` so the executor stamps
    `depth/space=metric` + `depth/unit=meters`
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from live_action_aov.io.channels import CH_DEPTH_CONFIDENCE, CH_Z, CH_Z_RAW  # noqa: E402
from live_action_aov.passes.depth.depthpro import DepthProPass  # noqa: E402


class _FakeDepthProOutput:
    """Mimics transformers' DepthEstimationOutput."""

    def __init__(self, depth: torch.Tensor, confidence: torch.Tensor | None) -> None:
        self.predicted_depth = depth
        self.confidence = confidence


class _FakeDepthPro(DepthProPass):
    """Injects a deterministic metric-depth field and confidence tensor."""

    def __init__(self, params=None, *, confidence: bool = True) -> None:
        super().__init__(params)
        self._emit_confidence = confidence

    def _load_model(self) -> None:  # type: ignore[override]
        if self._model is not None:
            return
        self._model = object()

        class _P:
            def __call__(self, images, return_tensors, do_resize, size, **kwargs):
                # Pretend we resized to a small fixed size; return a tensor
                # that's big enough for the bilinear upscale to do something.
                inf = 32
                img = np.asarray(images, dtype=np.float32)
                if img.ndim == 3:
                    img = img[None]
                t = torch.from_numpy(img).permute(0, 3, 1, 2)
                t = torch.nn.functional.interpolate(t, size=(inf, inf), mode="bilinear", align_corners=False)
                return {"pixel_values": t}

        self._processor = _P()
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def infer(self, tensor):  # type: ignore[override]
        pv = tensor["pixel_values"]
        b, _, h, w = pv.shape
        # Metric depth in meters — ramp from 2m (near) to 20m (far) across W.
        depth = torch.linspace(2.0, 20.0, w).view(1, 1, w).expand(b, h, w)
        depth = depth.clone()
        confidence = None
        if self._emit_confidence:
            # High confidence at image center, lower at edges.
            x = torch.linspace(-1.0, 1.0, w).abs()
            confidence = (1.0 - x).view(1, 1, w).expand(b, h, w).clone()
        return {
            "depth": depth,
            "confidence": confidence,
            "plate_shape": tensor["plate_shape"],
        }


class _FakeReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._first = 1

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._frames[idx - self._first], {}


def _flat_frame(h: int = 24, w: int = 32) -> np.ndarray:
    return np.full((h, w, 3), 0.5, dtype=np.float32)


def test_license_is_noncommercial_gated() -> None:
    lic = DepthProPass.declared_license()
    assert lic.commercial_use is False
    assert "Apple" in lic.notes


def test_run_shot_emits_metric_depth_and_confidence() -> None:
    frames = [_flat_frame() for _ in range(3)]
    pass_ = _FakeDepthPro()
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 3))

    assert sorted(out) == [1, 2, 3]
    for f in (1, 2, 3):
        assert {CH_Z, CH_Z_RAW, CH_DEPTH_CONFIDENCE} <= set(out[f])
        assert out[f][CH_Z].shape == (24, 32)
        assert out[f][CH_Z].dtype == np.float32
        # Metric depth — left edge ~2m, right edge ~20m (bilinear upscale).
        assert float(out[f][CH_Z][:, 0].mean()) == pytest.approx(2.0, abs=0.5)
        assert float(out[f][CH_Z][:, -1].mean()) == pytest.approx(20.0, abs=0.5)
        # Z_raw must equal Z for metric backend (same values, schema parity).
        assert np.array_equal(out[f][CH_Z], out[f][CH_Z_RAW])
        # Confidence clipped to [0, 1].
        c = out[f][CH_DEPTH_CONFIDENCE]
        assert c.min() >= 0.0 and c.max() <= 1.0
        # High at center, low at edges.
        mid = int(c.shape[1] // 2)
        assert float(c[:, mid].mean()) > float(c[:, 0].mean())


def test_no_per_clip_flip_normalization() -> None:
    """Metric backend emits raw meters — `Z` values outside [0, 1] are expected
    and we must NOT be applying the 1 - (d-min)/span trick."""
    frames = [_flat_frame() for _ in range(2)]
    pass_ = _FakeDepthPro()
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 2))
    z1 = out[1][CH_Z]
    # Values should be well outside [0,1] — they are meters.
    assert z1.max() > 5.0


def test_emit_artifacts_marks_depth_as_metric() -> None:
    frames = [_flat_frame() for _ in range(2)]
    pass_ = _FakeDepthPro()
    pass_.run_shot(_FakeReader(frames), frame_range=(1, 2))
    artifacts = pass_.emit_artifacts()
    assert "depth_metric" in artifacts
    # Not the same key as the relative backend uses.
    assert "depth_norm_min" not in artifacts
    assert "depth_norm_max" not in artifacts


def test_missing_confidence_falls_back_to_ones() -> None:
    frames = [_flat_frame() for _ in range(1)]
    pass_ = _FakeDepthPro(confidence=False)
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 1))
    conf = out[1][CH_DEPTH_CONFIDENCE]
    assert conf.shape == (24, 32)
    assert np.allclose(conf, 1.0)


def test_smoothable_channels_is_z_only() -> None:
    """Z should smooth, but Z_raw and confidence should not — smoothing raw
    metric signal or confidence would be noise propagation."""
    assert DepthProPass.smoothable_channels == [CH_Z]
