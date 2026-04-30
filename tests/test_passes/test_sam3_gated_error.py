"""Tests for the HF gated-repo error wrapper used by SAM3MattePass._load_model.

The matte pass downloads `facebook/sam3` on first use, which is gated.
Without the wrapper, users see a raw `OSError: You are trying to access
a gated repo.` with no actionable next step. The wrapper turns that into
a `RuntimeError` pointing at the three-step setup in `docs/install.md`.
"""

from __future__ import annotations

from live_action_aov.passes.matte.sam3 import _wrap_if_gated_repo

# Real-world payload — copied from the user-reported v0.1.1 traceback.
_GATED_OSERROR_TEXT = (
    "You are trying to access a gated repo.\n"
    "Make sure to have access to it at https://huggingface.co/facebook/sam3.\n"
    "401 Client Error. (Request ID: Root=1-69f370d1-786df2157a9688e95d9d0931)\n"
    "Cannot access gated repo for url "
    "https://huggingface.co/facebook/sam3/resolve/main/processor_config.json.\n"
    "Access to model facebook/sam3 is restricted. You must have access to it "
    "and be authenticated to access it. Please log in."
)


def test_wraps_real_world_oserror_traceback() -> None:
    err = OSError(_GATED_OSERROR_TEXT)
    wrapped = _wrap_if_gated_repo("facebook/sam3", err)
    assert wrapped is not None
    msg = str(wrapped)
    assert "facebook/sam3" in msg
    assert "https://huggingface.co/facebook/sam3" in msg
    assert "https://huggingface.co/settings/tokens" in msg
    assert "hf auth login" in msg


def test_wraps_minimal_gated_repo_substring() -> None:
    """Future HF versions may shorten the message — substring is the anchor."""
    err = OSError("Cannot access gated repo for url ...")
    assert _wrap_if_gated_repo("facebook/sam3", err) is not None


def test_wraps_401_message() -> None:
    err = OSError("Failed: 401 Client Error.")
    assert _wrap_if_gated_repo("facebook/sam3", err) is not None


def test_wraps_access_restricted_phrasing() -> None:
    err = OSError("Access to model facebook/sam3 is restricted.")
    assert _wrap_if_gated_repo("facebook/sam3", err) is not None


def test_passes_through_unrelated_oserror() -> None:
    """A network blip or disk error should NOT get the gated guidance message."""
    err = OSError("Connection timed out while downloading config.json")
    assert _wrap_if_gated_repo("facebook/sam3", err) is None


def test_passes_through_arbitrary_exception() -> None:
    err = ValueError("some unrelated bug")
    assert _wrap_if_gated_repo("facebook/sam3", err) is None


def test_repo_id_appears_verbatim_in_message() -> None:
    """Different gated models should each get their own URL in the guidance."""
    err = OSError("gated repo")
    wrapped = _wrap_if_gated_repo("meta-llama/Llama-3-8B", err)
    assert wrapped is not None
    assert "https://huggingface.co/meta-llama/Llama-3-8B" in str(wrapped)
