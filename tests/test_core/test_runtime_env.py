"""HF network defaults — set sane timeouts, never override the user."""

from __future__ import annotations

import os

from live_action_aov.core.runtime_env import apply_hf_network_defaults


def test_sets_defaults_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    apply_hf_network_defaults()
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "120"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "30"


def test_never_overrides_user_values(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_DOWNLOAD_TIMEOUT", "5")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "99")
    apply_hf_network_defaults()
    # setdefault must respect an explicit user override (incl. HF_HUB_OFFLINE
    # setups that deliberately tune these).
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "5"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "99"
