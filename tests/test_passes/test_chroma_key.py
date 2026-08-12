"""ChromaKeyPass — colour-difference pull bounded by SAM (no model, no fakes).

Unlike the ML refiners this pass is pure maths, so tests run the REAL
inference on synthetic screens: subject rectangle on a green/blue field.
"""

from __future__ import annotations

import numpy as np

from live_action_aov.io.channels import MASK_PREFIX
from live_action_aov.passes.matte.chroma_key import ChromaKeyPass


def _screen_plate(n: int, h: int, w: int, screen: str) -> np.ndarray:
    """Screen-coloured field with a grey subject rect + soft edge column."""
    rgb = np.zeros((n, h, w, 3), np.float32)
    if screen == "green":
        rgb[..., 1] = 0.7
        rgb[..., 0] = 0.1
        rgb[..., 2] = 0.1
    else:
        rgb[..., 2] = 0.7
        rgb[..., 0] = 0.1
        rgb[..., 1] = 0.1
    # Subject: neutral grey block.
    rgb[:, 8:24, 10:30, :] = 0.4
    # A soft-edge column: 50/50 mix of subject and screen at x=30.
    rgb[:, 8:24, 30, :] = 0.5 * rgb[:, 8:24, 29, :] + 0.5 * rgb[:, 8:24, 31, :]
    return rgb


def _hard(n: int, h: int, w: int) -> np.ndarray:
    m = np.zeros((n, h, w), np.float32)
    m[:, 8:24, 10:31] = 1.0
    return m


def _run(screen_plate: np.ndarray, hard: np.ndarray, **params) -> np.ndarray:
    p = ChromaKeyPass(params or {})
    return p._refine_instance(screen_plate, hard)


def test_green_screen_pull() -> None:
    n, h, w = 2, 32, 48
    soft = _run(_screen_plate(n, h, w, "green"), _hard(n, h, w))
    assert soft[0, 16, 20] == 1.0  # subject core solid
    assert soft[0, 4, 40] == 0.0  # screen far outside the bound
    # Inside the bound but on pure screen (dilated ring): key kills it.
    assert soft[0, 16, 40] < 0.1


def test_blue_screen_pull_auto_detected() -> None:
    n, h, w = 2, 32, 48
    soft = _run(_screen_plate(n, h, w, "blue"), _hard(n, h, w))  # screen="auto"
    assert soft[0, 16, 20] == 1.0
    assert soft[0, 16, 40] < 0.1


def test_soft_edge_pixel_is_semi_transparent() -> None:
    """The 50/50 mixed column must land strictly between 0 and 1 — the
    physical semi-transparency a key gives and a hard mask can't."""
    n, h, w = 1, 32, 48
    soft = _run(_screen_plate(n, h, w, "blue"), _hard(n, h, w))
    edge = float(soft[0, 16, 30])
    assert 0.05 < edge < 0.95, f"edge alpha {edge} not semi-transparent"


def test_screen_coloured_clothing_cannot_hole_the_core() -> None:
    """Blue jeans on a blue screen: screen-coloured pixels INSIDE the SAM
    core must stay solid 1.0 (the core guarantee)."""
    n, h, w = 1, 32, 48
    plate = _screen_plate(n, h, w, "blue")
    plate[:, 14:18, 14:18, :] = [0.1, 0.1, 0.7]  # blue patch inside subject
    soft = _run(plate, _hard(n, h, w), screen="blue")
    assert soft[0, 16, 16] == 1.0, "screen-coloured clothing holed the matte"


def test_forced_screen_overrides_auto() -> None:
    n, h, w = 1, 32, 48
    plate = _screen_plate(n, h, w, "green")
    soft = _run(plate, _hard(n, h, w), screen="green")
    assert soft[0, 16, 40] < 0.1  # pure screen inside the bound is killed


def test_emits_standard_refiner_channels() -> None:
    """Inherits the refiner contract: matte.rgba heroes + mask.<label>."""
    n, h, w = 2, 32, 48

    class Reader:
        def __init__(self, frames):
            self.f = frames

        def read_frame(self, i):
            return self.f[i - 1], {}

    plate = _screen_plate(n, h, w, "green")
    hard = _hard(n, h, w)
    stack = (hard > 0.5).astype(np.uint8)
    art = {
        "sam3_hard_masks": {0: {1: {"label": "hero", "frames": [1, 2], "stack": stack}}},
        "sam3_instances": {0: [
            {"track_id": 1, "slot": "r", "label": "hero", "score": 0.9, "frames": [1, 2]}
        ]},
    }
    p = ChromaKeyPass({"refine_all_masks": True})
    p.ingest_artifacts(art)
    out = p.run_shot(Reader(list(plate)), (1, 2))
    assert out[1]["matte.r"][16, 20] == 1.0
    assert out[1][f"{MASK_PREFIX}hero"][16, 20] == 1.0


def test_license_is_unencumbered() -> None:
    lic = ChromaKeyPass.declared_license()
    assert lic.commercial_use is True and lic.spdx == "MIT"


def test_desaturated_screen_still_keys_to_zero() -> None:
    """The field bug: display transforms (AgX) desaturate the screen, the
    colour-difference halves, and a fixed gain under-pulled — the screen
    stayed at alpha ~0.5 inside the dilated bound (visible halo around the
    subject). The per-frame calibration must kill a washed-out screen."""
    n, h, w = 1, 32, 48
    rgb = np.zeros((n, h, w, 3), np.float32)
    # AgX-ish pastel green: weak colour difference (d = 0.45 - 0.25 = 0.2).
    rgb[..., 0] = 0.25
    rgb[..., 1] = 0.45
    rgb[..., 2] = 0.25
    rgb[:, 8:24, 10:30, :] = 0.4  # neutral subject
    soft = _run(rgb, _hard(n, h, w))
    assert soft[0, 16, 20] == 1.0  # subject solid
    # Screen INSIDE the dilated bound — the halo region — must be dead.
    assert soft[0, 16, 40] < 0.05, f"washed-out screen leaked: {soft[0, 16, 40]}"


def test_non_screen_neighbour_stays_out_of_the_band() -> None:
    """Field bug: where the subject borders a NON-screen area (a car), the
    key kept everything in the dilated band and the matte swallowed the
    neighbour. Non-screen band pixels must follow SAM's hard edge (0
    outside the mask), while screen-side edges stay key-soft."""
    n, h, w = 1, 48, 64
    plate = _screen_plate(n, h, w, "green")
    # A grey "car" region to the RIGHT of the subject, outside the mask.
    plate[:, 8:24, 34:60, :] = 0.25
    soft = _run(plate, _hard(n, h, w))
    # Inside the band over the CAR (non-screen, outside SAM) -> excluded.
    assert soft[0, 16, 38] == 0.0, f"neighbour swallowed: {soft[0, 16, 38]}"
    # Subject core still solid; screen side (above subject) still keyed out.
    assert soft[0, 16, 20] == 1.0
    assert soft[0, 2, 20] < 0.05
