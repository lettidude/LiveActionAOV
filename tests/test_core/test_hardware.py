"""Hardware detection, proposed settings and long-clip windows (`core.hardware`)."""

from __future__ import annotations

import itertools

from live_action_aov.core import hardware as hwm
from live_action_aov.core.hardware import Hardware

UHD = (3840, 2160)


def test_detect_never_raises_and_reads_ram() -> None:
    hw = hwm.detect()
    assert hw.ram_gb is None or hw.ram_gb > 0
    assert isinstance(hw.summary(), str)


def test_tiers_follow_the_card() -> None:
    def tier(vram: float | None) -> tuple[str, int | None]:
        s = hwm.suggest(Hardware("card", vram, 128.0, 100.0), UHD, 24)
        return s.tier, s.proxy_long_edge

    assert tier(32.0) == ("quality", 2816)
    assert tier(24.0) == ("balanced", 1920)
    assert tier(12.0) == ("small", 1280)
    assert tier(8.0) == ("minimal", 960)
    assert tier(None)[0] == "none"
    # A plate smaller than the tier's proxy runs at full size.
    hd = hwm.suggest(Hardware("card", 32.0, 128.0, 100.0), (1920, 1080), 24)
    assert hd.proxy_long_edge is None


def test_sam31_only_when_installed_and_the_card_has_the_memory() -> None:
    big, mid = Hardware("a", 32.0, 128.0, 100.0), Hardware("b", 16.0, 128.0, 100.0)
    assert hwm.suggest(big, UHD, 24, sam31_installed=True).detector == "sam3.1"
    assert hwm.suggest(big, UHD, 24, sam31_installed=False).detector == "sam3"
    assert hwm.suggest(mid, UHD, 24, sam31_installed=True).detector == "sam3"


def test_ram_estimate_grows_with_frames_and_proxy() -> None:
    small = hwm.estimate_ram_gb(UHD, 100, 1920)
    big = hwm.estimate_ram_gb(UHD, 100, 2816)
    assert hwm.BASE_RAM_GB < small < big
    per_100 = small - hwm.BASE_RAM_GB
    assert abs(hwm.estimate_ram_gb(UHD, 200, 1920) - small - per_100) < 1e-6


def test_windows_only_when_the_clip_does_not_fit() -> None:
    assert hwm.plan_windows(UHD, 100, 128.0, 1920) is None  # fits
    assert hwm.plan_windows(UHD, 100, None, 1920) is None  # RAM unknown: one go
    win = hwm.plan_windows(UHD, 600, 64.0, 1920)
    assert win is not None and hwm.MIN_WINDOW <= win < 600
    # The window plus its two overlaps fits the budget it was sized from.
    run = win + 2 * hwm.WINDOW_OVERLAP
    assert hwm.estimate_ram_gb(UHD, run, 1920) <= 0.6 * 64.0 + 0.5


def test_long_clip_steps_the_proxy_down_before_tiny_windows() -> None:
    hw = Hardware("card", 32.0, 94.0, 69.0)
    s = hwm.suggest(hw, (4096, 1936), 100)
    assert s.proxy_long_edge == 1920 and s.window_frames is None
    assert any("proxy 1920 instead" in n for n in s.notes)
    short = hwm.suggest(hw, (4096, 1936), 40)
    assert short.proxy_long_edge == 2816 and short.window_frames is None
    long = hwm.suggest(hw, (4096, 1936), 600)
    assert long.proxy_long_edge == 1920 and long.window_frames is not None
    assert "windows" in long.one_line()


def test_window_spans_tile_the_clip_with_context() -> None:
    spans = hwm.window_spans(100, 40, 16)
    assert [(c0, c1) for _, _, c0, c1 in spans] == [(0, 40), (40, 80), (80, 100)]
    assert spans[0][:2] == (0, 56) and spans[1][:2] == (24, 96) and spans[2][:2] == (64, 100)
    assert hwm.window_spans(30, 40) == [(0, 30, 0, 30)]
    # A short tail joins the window before it instead of being a window of its own.
    tail = hwm.window_spans(85, 40, 16)
    assert [(c0, c1) for _, _, c0, c1 in tail] == [(0, 40), (40, 85)]
    for n, w in ((100, 40), (85, 40), (586, 111), (7, 3)):
        cores = [(c0, c1) for _, _, c0, c1 in hwm.window_spans(n, w, 16)]
        assert cores[0][0] == 0 and cores[-1][1] == n
        assert all(a[1] == b[0] for a, b in itertools.pairwise(cores))
