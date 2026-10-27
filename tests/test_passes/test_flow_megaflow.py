"""MegaFlow pass — plumbing with the model faked (no weights, no GPU).

The pass subclasses RAFTPass: only loading / preprocessing / pair inference
are its own. We fake the model with a callable that returns a known constant
flow in MegaFlow's output layout ((B, T-1, 2, h, w) in `flow_preds`) and
check the pair contract end to end: 14-multiple inference size, [0,255]
input scaling, backward pass via time-flip, vector scaling on the upscale,
confidence in [0,1], and run_shot's per-frame assembly.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from live_action_aov.io.channels import (  # noqa: E402
    CH_BACK_X,
    CH_FLOW_CONFIDENCE,
    CH_FLOW_OCCLUSION,
    CH_MOTION_X,
    CH_MOTION_Y,
)
from live_action_aov.passes.flow.megaflow import MegaFlowPass  # noqa: E402


class _FakeMegaFlow:
    """Returns +3 px in x at inference res for every pair; records inputs."""

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, video: torch.Tensor, num_reg_refine: int = 1) -> dict:
        self.calls.append(video)
        b, t, _c, h, w = video.shape
        f = torch.zeros(b, t - 1, 2, h, w)
        f[:, :, 0] = 3.0
        return {"flow_preds": [f]}


def _armed(params: dict | None = None) -> tuple[MegaFlowPass, _FakeMegaFlow]:
    p = MegaFlowPass({"inference_resolution": 56, **(params or {})})
    fake = _FakeMegaFlow()
    p._model = fake
    p._device = torch.device("cpu")
    p._dtype = torch.float32
    return p, fake


def test_preprocess_rounds_to_patch_and_scales_to_255() -> None:
    p, _ = _armed()
    pair = np.full((2, 60, 100, 3), 0.5, np.float32)
    t = p.preprocess(pair)
    assert t["inf_shape"] == (28, 56)  # long edge 56; 60*0.56=33.6 -> nearest /14 = 28
    assert tuple(t["video"].shape) == (1, 2, 3, 28, 56)
    assert abs(float(t["video"].mean()) - 127.5) < 1.0  # [0,1] -> [0,255]
    assert t["plate_shape"] == (60, 100)


def test_pair_flow_is_upscaled_with_vector_scaling_and_backward_is_time_flipped() -> None:
    p, fake = _armed()
    pair = np.random.default_rng(0).uniform(0.2, 0.8, (2, 60, 100, 3)).astype(np.float32)
    out = p.postprocess(p.infer(p.preprocess(pair)))
    # +3 px at 56 wide -> +3 * (100/56) at plate width.
    assert out[CH_MOTION_X].shape == (60, 100)
    assert abs(float(np.median(out[CH_MOTION_X])) - 3.0 * 100 / 56) < 0.05
    assert abs(float(np.median(out[CH_MOTION_Y]))) < 1e-4
    # Backward pass fed the time-flipped video (frame b first).
    assert len(fake.calls) == 2
    assert torch.equal(fake.calls[1][:, 0], fake.calls[0][:, 1])
    assert float(np.median(out[CH_BACK_X])) > 0  # fake is +3 for any order
    conf = out[CH_FLOW_CONFIDENCE]
    assert conf.min() >= 0.0 and conf.max() <= 1.0
    assert np.allclose(out[CH_FLOW_OCCLUSION], 1.0 - conf)


def test_run_shot_assembles_per_frame_channels() -> None:
    p, _ = _armed()
    frames = [np.full((28, 56, 3), 0.4, np.float32) for _ in range(3)]

    class _R:
        def read_frame(self, i: int):
            return frames[i - 1], {}

    out = p.run_shot(_R(), (1, 3))
    assert set(out) == {1, 2, 3}
    assert abs(float(np.median(out[1][CH_MOTION_X])) - 3.0) < 1e-3  # 56 wide = no rescale
    assert float(np.abs(out[3][CH_MOTION_X]).max()) == 0.0  # last frame: no forward pair
    assert float(np.abs(out[1][CH_BACK_X]).max()) == 0.0  # first frame: no backward pair
    arts = p.emit_artifacts()
    assert "forward_flow" in arts and "parallax_estimate" in arts


def test_registered_commercial_clean() -> None:
    lic = MegaFlowPass.declared_license()
    assert lic.spdx == "Apache-2.0" and lic.commercial_use is True
    assert MegaFlowPass.name == "flow_megaflow"
