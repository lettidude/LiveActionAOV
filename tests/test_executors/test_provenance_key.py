"""Nuke's Cryptomatte node scans every EXR header key for the substring
``cryptomatte/`` and treats the text before it as a layer id. A provenance
key such as ``liveaov/cryptomatte/commercial`` crashed Nuke 13.2v6 (segfault
on evaluation, bisected to those four keys). No
provenance key may contain that substring."""

from live_action_aov.executors.local import METADATA_NAMESPACE, provenance_key


def test_cryptomatte_pass_key_has_no_cryptomatte_slash() -> None:
    key = f"{METADATA_NAMESPACE}/{provenance_key('cryptomatte')}/commercial"
    assert "cryptomatte/" not in key
    assert key == f"{METADATA_NAMESPACE}/cryptomatte_pass/commercial"


def test_other_passes_are_untouched() -> None:
    for name in ("sam3_matte", "moge3", "flow_megaflow", "vitmatte_refiner"):
        assert provenance_key(name) == name


def test_keys_never_collide() -> None:
    # a registered pass called `cryptomatte_pass` next to the built-in `cryptomatte`
    names = [
        "cryptomatte",
        "cryptomatte_pass",
        "cryptomatte_pass_pass",
        "my_cryptomatte",
        "x_pass",
        "x",
    ]
    keys = [provenance_key(n) for n in names]
    assert len(set(keys)) == len(keys)
    assert not any("cryptomatte/" in f"{k}/" for k in keys)
