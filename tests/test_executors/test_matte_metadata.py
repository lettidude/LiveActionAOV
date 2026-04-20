"""Executor writes `liveaov/matte/*` metadata when a matte pipeline runs.

Wires fake detector + fake refiner into the registry (both commercial-safe,
so the license gate is inactive) and asserts the sidecar has:
- `matte/detector`  / `matte/refiner`
- `matte/commercial = "true"`  (refiner is MIT)
- `matte/concepts`  (from detector's `matte_concepts` artifact)
- `matte/hero_r` / `hero_g` (label, track_id, score triples)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("torch")

from live_action_aov.cli.app import app
from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.core.registry import get_registry
from live_action_aov.io.channels import (
    CH_MATTE_A,
    CH_MATTE_B,
    CH_MATTE_G,
    CH_MATTE_R,
)
from live_action_aov.io.oiio_io import HAS_OIIO

runner = CliRunner()


class _FakeMatteDetector(UtilityPass):
    """Fakes SAM3: emits mask.person channel + the three matte artifacts."""

    name = "fake_matte_detector"
    version = "0.0.1"
    license = License(spdx="SAM-License-1.0", commercial_use=True, notes="Test-only")
    pass_type = PassType.SEMANTIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    produces_channels: list[ChannelSpec] = []
    provides_artifacts = ["sam3_hard_masks", "sam3_instances", "matte_concepts"]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._frame_keys: list[int] = []

    def preprocess(self, frames: np.ndarray):
        return frames

    def infer(self, tensor: Any):
        return tensor

    def postprocess(self, tensor: Any):
        return {}

    def run_shot(self, reader: Any, frame_range: tuple[int, int]):  # type: ignore[override]
        first, last = frame_range
        self._frame_keys = list(range(first, last + 1))
        out: dict[int, dict[str, np.ndarray]] = {}
        # One frame read to discover plate size.
        sample, _ = reader.read_frame(first)
        h, w = sample.shape[:2]
        self._plate_shape = (h, w)
        mask = np.ones((h, w), dtype=np.float32) * 0.8
        for f in range(first, last + 1):
            out[f] = {"mask.person": mask.copy()}
        return out

    def emit_artifacts(self):  # type: ignore[override]
        if not self._frame_keys:
            return {}
        h, w = self._plate_shape
        stack = np.ones((len(self._frame_keys), h, w), dtype=np.float32) * 0.8
        hard = {
            42: {
                "label": "person",
                "frames": list(self._frame_keys),
                "stack": stack,
            }
        }
        heroes = [
            {
                "track_id": 42,
                "slot": "r",
                "label": "person",
                "score": 0.91,
                "frames": list(self._frame_keys),
            }
        ]
        any_frame = self._frame_keys[0]
        return {
            "sam3_hard_masks": {any_frame: hard},
            "sam3_instances": {any_frame: heroes},
            "matte_concepts": {any_frame: ["person"]},
        }


class _FakeRefiner(UtilityPass):
    """Fakes RVM: packs hero into matte.r and publishes matte_heroes."""

    name = "fake_refiner"
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only")
    pass_type = PassType.SEMANTIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    produces_channels = [
        ChannelSpec(name=CH_MATTE_R),
        ChannelSpec(name=CH_MATTE_G),
        ChannelSpec(name=CH_MATTE_B),
        ChannelSpec(name=CH_MATTE_A),
    ]
    requires_artifacts = ["sam3_hard_masks", "sam3_instances"]
    provides_artifacts = ["matte_heroes"]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._heroes: list[dict[str, Any]] = []
        self._frame_keys: list[int] = []

    def ingest_artifacts(self, artifacts):  # type: ignore[override]
        heroes = artifacts.get("sam3_instances") or {}
        if heroes:
            self._heroes = list(next(iter(heroes.values())) or [])

    def preprocess(self, frames: np.ndarray):
        return frames

    def infer(self, tensor: Any):
        return tensor

    def postprocess(self, tensor: Any):
        return {}

    def run_shot(self, reader: Any, frame_range: tuple[int, int]):  # type: ignore[override]
        first, last = frame_range
        self._frame_keys = list(range(first, last + 1))
        sample, _ = reader.read_frame(first)
        h, w = sample.shape[:2]
        z = np.zeros((h, w), dtype=np.float32)
        matte_r = np.full((h, w), 0.5, dtype=np.float32)
        out: dict[int, dict[str, np.ndarray]] = {}
        for f in self._frame_keys:
            out[f] = {
                CH_MATTE_R: matte_r.copy(),
                CH_MATTE_G: z.copy(),
                CH_MATTE_B: z.copy(),
                CH_MATTE_A: z.copy(),
            }
        return out

    def emit_artifacts(self):  # type: ignore[override]
        if not self._heroes:
            return {}
        return {
            "matte_heroes": {
                self._frame_keys[0]: [
                    {
                        "track_id": h["track_id"],
                        "slot": h["slot"],
                        "label": h["label"],
                        "score": h["score"],
                        "refined_frames": list(h.get("frames", [])),
                        "missing_hard_mask": False,
                    }
                    for h in self._heroes
                ]
            }
        }


@pytest.fixture(autouse=True)
def _register_fakes() -> None:
    reg = get_registry()
    reg.register_pass("fake_matte_detector", _FakeMatteDetector)
    reg.register_pass("fake_refiner", _FakeRefiner)


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_sidecar_has_matte_metadata_and_commercial_flag(test_plate_1080p: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run-shot",
            str(test_plate_1080p),
            "--passes",
            "fake_matte_detector,fake_refiner",
        ],
    )
    assert result.exit_code == 0, result.stdout

    from live_action_aov.io.oiio_io import read_exr

    sidecar = sorted(test_plate_1080p.glob("test_plate.utility.*.exr"))[0]
    _, attrs = read_exr(sidecar)

    # Detector + refiner identity attributes.
    assert attrs.get("liveaov/matte/detector") == "fake_matte_detector"
    assert attrs.get("liveaov/matte/refiner") == "fake_refiner"
    # Commercial flag — MIT refiner → "true" (string, not bool, for Nuke).
    assert attrs.get("liveaov/matte/commercial") == "true"
    # Concept list (comma-joined) from matte_concepts artifact.
    assert attrs.get("liveaov/matte/concepts") == "person"
    # Hero metadata for slot r — label, track_id, score all present.
    assert attrs.get("liveaov/matte/hero_r/label") == "person"
    assert int(attrs.get("liveaov/matte/hero_r/track_id")) == 42
    assert float(attrs.get("liveaov/matte/hero_r/score")) == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# Round 2: non-commercial refiner → `matte/commercial = "false"`
# ---------------------------------------------------------------------------


class _FakeNCRefiner(_FakeRefiner):
    """Same as `_FakeRefiner` but with a non-commercial license, mirroring
    MatAnyone 2's NTU-S-Lab-1.0. The executor must emit
    `matte/commercial = "false"` so downstream QC catches the deliverable
    before it ships to a commercial client."""

    name = "fake_nc_refiner"
    license = License(
        spdx="NTU-S-Lab-1.0",
        commercial_use=False,
        commercial_tool_resale=False,
        notes="Test-only NC license",
    )


@pytest.fixture
def _register_nc_refiner() -> None:
    get_registry().register_pass("fake_nc_refiner", _FakeNCRefiner)


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_sidecar_commercial_flag_is_false_for_noncommercial_refiner(
    test_plate_1080p: Path,
    _register_nc_refiner: None,
) -> None:
    """Mirror of the round-1 test but with an NC refiner. Everything else
    must stay identical — only `matte/commercial` flips to `"false"`.
    Detector stays commercial-safe (SAM 3 carve-out) so this purely
    exercises the "refiner license decides matte commercial flag" rule."""
    result = runner.invoke(
        app,
        [
            "run-shot",
            str(test_plate_1080p),
            "--passes",
            "fake_matte_detector,fake_nc_refiner",
            "--allow-noncommercial",  # gate must be open for NC refiner
        ],
    )
    assert result.exit_code == 0, result.stdout

    from live_action_aov.io.oiio_io import read_exr

    sidecar = sorted(test_plate_1080p.glob("test_plate.utility.*.exr"))[0]
    _, attrs = read_exr(sidecar)

    assert attrs.get("liveaov/matte/refiner") == "fake_nc_refiner"
    assert attrs.get("liveaov/matte/commercial") == "false"
    # Detector identity + hero metadata still come through.
    assert attrs.get("liveaov/matte/detector") == "fake_matte_detector"
    assert attrs.get("liveaov/matte/hero_r/label") == "person"
