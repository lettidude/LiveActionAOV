"""Older Nuke Cryptomatte nodes crash on any NON-string header attribute whose
name contains "cryptomatte" (`liveaov/cryptomatte/commercial`, then
`liveaov/cryptomatte_pass/commercial`). No provenance key may contain the
word; the cryptomatte pass is recorded under `liveaov/crypto_pass/...`."""

from live_action_aov.executors.local import METADATA_NAMESPACE, provenance_key, provenance_keys


def test_cryptomatte_pass_key_has_no_cryptomatte_in_it() -> None:
    key = f"{METADATA_NAMESPACE}/{provenance_key('cryptomatte')}/commercial"
    assert "cryptomatte" not in key
    assert key == f"{METADATA_NAMESPACE}/crypto_pass/commercial"


def test_other_passes_are_untouched() -> None:
    for name in ("sam3_matte", "moge3", "flow_megaflow", "vitmatte_refiner"):
        assert provenance_key(name) == name


def test_keys_never_collide() -> None:
    # registered passes with contrived names next to the built-in `cryptomatte`
    names = [
        "cryptomatte",
        "cryptomatte_pass",
        "cryptomatte_pass_pass",
        "crypto",
        "crypto_pass",
        "my_cryptomatte",
        "x_pass",
        "x",
    ]
    keys = provenance_keys(names)
    assert len(set(keys.values())) == len(names)
    assert not any("cryptomatte" in k for k in keys.values())
    assert keys["cryptomatte"] == "crypto_pass"
