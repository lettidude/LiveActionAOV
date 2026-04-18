"""DSINEPass — contract tests with a fake torch.hub model.

The real `torch.hub.load('baegwangbin/DSINE', ...)` pulls weights over the
network and is too heavy for CI. We inject a fake model that returns known
tensors so we can verify the actual contract:
  - unit-length normals after upscale + renormalize (spec §11.3 trap 2)
  - intrinsics scaling by inference/plate ratio (trap 3)
  - axis conversion opencv → opengl (spec §10.3)
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from live_action_aov.io.channels import CH_N_X, CH_N_Y, CH_N_Z  # noqa: E402
from live_action_aov.passes.normals.dsine import (  # noqa: E402
    DSINEPass,
    _convert_axes,
    _scaled_intrinsics,
)


class _FakeDSINEModel(torch.nn.Module):
    """Returns a horizontal ramp for N.x and mostly +Z normals — good enough to
    probe the post-processing contract."""

    def forward(self, image: torch.Tensor, intrins: torch.Tensor | None = None) -> torch.Tensor:
        b, _, h, w = image.shape
        ramp = torch.linspace(-0.9, 0.9, w).view(1, 1, 1, w).expand(b, 1, h, w)
        nz = torch.ones(b, 1, h, w)
        ny = torch.zeros(b, 1, h, w)
        raw = torch.cat([ramp, ny, nz], dim=1)
        # Return as a list — DSINE's real forward returns a multi-scale list.
        return [raw]


class _FakePass(DSINEPass):
    def _load_model(self) -> None:  # type: ignore[override]
        if self._model is not None:
            return
        self._model = _FakeDSINEModel().eval()
        self._device = torch.device("cpu")
        self._dtype = torch.float32


def test_license_is_commercial_safe() -> None:
    lic = DSINEPass.declared_license()
    assert lic.spdx == "MIT"
    assert lic.commercial_use is True


def test_normals_are_unit_length_at_plate_resolution() -> None:
    pass_ = _FakePass({"output_axes": "opencv"})  # skip axis flip to isolate the trap-2 fix
    plate = np.full((1, 64, 96, 3), 0.5, dtype=np.float32)
    model_in = pass_.preprocess(plate)
    model_out = pass_.infer(model_in)
    ch = pass_.postprocess(model_out)

    assert ch[CH_N_X].shape == (64, 96)
    assert ch[CH_N_X].dtype == np.float32

    # sqrt(Nx^2 + Ny^2 + Nz^2) ≈ 1 everywhere (trap 2 check).
    mag = np.sqrt(ch[CH_N_X] ** 2 + ch[CH_N_Y] ** 2 + ch[CH_N_Z] ** 2)
    assert np.allclose(mag, 1.0, atol=1e-3)


def test_axis_convention_opencv_to_opengl_flips_y_and_z() -> None:
    """Default DSINE config outputs opengl convention. Fake fwd emits +Z, +0
    Y, so after flip we should see -Z (from +Z flipped)."""
    pass_ = _FakePass()  # default: opencv → opengl
    plate = np.full((1, 32, 32, 3), 0.5, dtype=np.float32)
    out = pass_.postprocess(pass_.infer(pass_.preprocess(plate)))

    # Fake Nz was +1 in OpenCV. Flipping to OpenGL negates it.
    # After renormalize, Nz ≈ -1 / norm, norm depends on Nx ramp but stays <= 1.
    assert float(out[CH_N_Z].mean()) < 0.0


def test_convert_axes_helper_is_involutory() -> None:
    rng = np.random.default_rng(0)
    n = rng.standard_normal((3, 5, 7)).astype(np.float32)
    there = _convert_axes(n, src="opencv", dst="opengl")
    back = _convert_axes(there, src="opengl", dst="opencv")
    assert np.allclose(back, n)


def test_scaled_intrinsics_match_resize_ratio() -> None:
    """fx,fy,cx,cy should scale by inf/plate on each axis (spec §11.3 trap 3)."""
    plate_h, plate_w = 1080, 1920
    inf_h, inf_w = 540, 960          # exactly 0.5x on each axis
    K = _scaled_intrinsics(
        plate_h, plate_w, inf_h, inf_w,
        fx=1000.0, fy=1000.0, cx=960.0, cy=540.0,
    )
    K_np = K[0].cpu().numpy()
    # Both axes scaled by 0.5 → fx=500, fy=500, cx=480, cy=270.
    assert K_np[0, 0] == pytest.approx(500.0)
    assert K_np[1, 1] == pytest.approx(500.0)
    assert K_np[0, 2] == pytest.approx(480.0)
    assert K_np[1, 2] == pytest.approx(270.0)
    assert K_np[2, 2] == pytest.approx(1.0)


def test_default_intrinsics_use_50mm_equivalent_heuristic() -> None:
    """When no shot intrinsics are present, fall back to f = 0.8 * max(W, H)
    at plate res, then scale to inference res (design §8.4).
    """
    plate_h, plate_w = 1000, 1000
    inf_h, inf_w = 500, 500          # 0.5x
    K = _scaled_intrinsics(
        plate_h, plate_w, inf_h, inf_w,
        fx=None, fy=None, cx=None, cy=None,
    )
    K_np = K[0].cpu().numpy()
    # Plate fx = 0.8 * 1000 = 800; scaled 0.5 → 400.
    assert K_np[0, 0] == pytest.approx(400.0)
    assert K_np[1, 1] == pytest.approx(400.0)
    # Plate cx = 500; scaled 0.5 → 250.
    assert K_np[0, 2] == pytest.approx(250.0)
    assert K_np[1, 2] == pytest.approx(250.0)
