"""FlowCache basic put/get/has semantics."""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.shared.optical_flow.cache import FlowCache


def test_put_get_roundtrip() -> None:
    cache = FlowCache()
    flow = np.ones((2, 4, 4), dtype=np.float32)
    cache.put("shotA", 3, "forward", flow)
    assert cache.has("shotA", 3, "forward")
    assert not cache.has("shotA", 3, "backward")
    assert cache.get("shotA", 3, "forward") is flow or np.array_equal(
        cache.get("shotA", 3, "forward"), flow
    )


def test_get_missing_returns_none() -> None:
    cache = FlowCache()
    assert cache.get("shotA", 99, "forward") is None


def test_rejects_wrong_shape() -> None:
    cache = FlowCache()
    with pytest.raises(ValueError):
        cache.put("s", 1, "forward", np.zeros((3, 4, 4), dtype=np.float32))


def test_clear_per_shot() -> None:
    cache = FlowCache()
    cache.put("A", 1, "forward", np.zeros((2, 4, 4), dtype=np.float32))
    cache.put("B", 1, "forward", np.zeros((2, 4, 4), dtype=np.float32))
    cache.clear("A")
    assert not cache.has("A", 1, "forward")
    assert cache.has("B", 1, "forward")
