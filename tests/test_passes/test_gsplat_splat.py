"""Gaussian splat pass — helpers without a GPU, plus a gated GPU smoke test.

The CPU tests pin the parts that silently break comp: the colour encoding
round-trip (the clean plate must comp back over the plate), the GL->OpenCV
view matrix (splat and Nuke camera must agree), SAM 3 masks excluding
pixels from training, and the 3DGS PLY layout viewers expect.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from live_action_aov.passes.scene.gsplat_splat import (
    GsplatSplatPass,
    _frame_mask,
    _object_masks,
    _viewmat,
    decode,
    encode,
    write_3dgs_ply,
)


def test_encode_decode_roundtrip_keeps_hdr():
    x = np.array([0.0, 1e-4, 0.0031308, 0.18, 1.0, 4.0, 16.0], np.float32)
    assert np.allclose(decode(encode(x)), x, rtol=1e-5, atol=1e-7)
    assert encode(np.array([16.0]))[0] > 1.0  # highlights are not clipped


def _ext(R: np.ndarray, t: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(rotation_matrix=R.tolist(), translation=t.tolist(), frame=1)


def test_viewmat_matches_gl_camera_projection():
    """A point projected with the gsplat (OpenCV) view matrix lands on the
    same pixel as through the Nuke-style GL camera."""
    a = math.radians(20)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    t = np.array([0.5, 0.2, 3.0])
    V = _viewmat(_ext(R, t)).astype(np.float64)
    f, cx, cy = 1000.0, 960.0, 540.0
    X = np.array([1.0, -0.3, -4.0])
    c_gl = R.T @ (X - t)
    gl = (cx + f * c_gl[0] / -c_gl[2], cy - f * c_gl[1] / -c_gl[2])
    c_cv = V[:3, :3] @ X + V[:3, 3]
    cv = (cx + f * c_cv[0] / c_cv[2], cy + f * c_cv[1] / c_cv[2])
    assert c_cv[2] > 0  # in front of the camera
    assert np.allclose(gl, cv)


def test_sam3_objects_are_excluded_from_training():
    stack = np.zeros((1, 20, 40), np.uint8)
    stack[0, 5:10, 10:20] = 1
    tracks = _object_masks(
        {3: {"label": "person", "frames": [7], "stack": stack}}, {"mask_objects": True}
    )
    m = _frame_mask(tracks, 7, (40, 80), {"mask_dilate_frac": 0.0})
    assert m.shape == (40, 80) and m.dtype == np.float32
    assert m[15, 30] == 0.0 and m[0, 0] == 1.0  # object out, background in
    assert (_frame_mask(tracks, 8, (40, 80), {}) == 1.0).all()  # not on this frame
    assert (
        _object_masks(
            {3: {"label": "person", "frames": [7], "stack": stack}}, {"mask_labels": ["car"]}
        )
        == []
    )


def test_3dgs_ply_layout(tmp_path: Path):
    n = 5
    splats = {
        "means": np.random.rand(n, 3).astype(np.float32),
        "scales": np.zeros((n, 3), np.float32),
        "quats": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        "opacities": np.zeros(n, np.float32),
        "sh0": np.zeros((n, 1, 3), np.float32),
        "shN": np.arange(n * 15 * 3, dtype=np.float32).reshape(n, 15, 3),
    }
    p = tmp_path / "s.ply"
    write_3dgs_ply(splats, p)
    raw = p.read_bytes()
    head, body = raw.split(b"end_header\n", 1)
    props = [ln.split()[-1].decode() for ln in head.splitlines() if ln.startswith(b"property")]
    assert props[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    assert props[-8:] == [
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    assert len(props) == 62 and b"element vertex 5" in head
    data = np.frombuffer(body, "<f4").reshape(n, 62)
    # f_rest is channel-major (INRIA): first 15 values = red of every SH band.
    assert np.allclose(data[0, 9:24], splats["shN"][0, :, 0])


def test_missing_camera_solve_is_actionable():
    pytest.importorskip("gsplat")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    p = GsplatSplatPass()
    p.ingest_artifacts({})
    with pytest.raises(RuntimeError, match="Camera track"):
        p.run_shot(None, (1, 3))


def test_contract():
    assert GsplatSplatPass.requires_artifacts == ["camera_solve"]
    assert GsplatSplatPass.optional_artifacts == ["sam3_hard_masks"]
    assert GsplatSplatPass.input_colorspace == "scene_linear"
    assert GsplatSplatPass.declared_license().commercial_use is True
