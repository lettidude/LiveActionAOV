# Cryptomatte (per-object IDs)

Turns SAM 3's tracked instances into an industry-standard **Cryptomatte** —
click any object in Nuke and get a perfect, temporally-stable matte. The
killer comp feature: no manual roto per object.

## How it works

`cryptomatte` is a dependent pass (same shape as the RVM refiner): it
consumes the `sam3_hard_masks` artifact the **SAM 3** detector publishes
(`{track_id: {label, frames, stack}}`) and encodes it to the Psyop
Cryptomatte spec. Each track becomes an object `<label>_<track_id>`
(`person_1`, `vehicle_2`, …); SAM 3's track IDs are temporally consistent,
so the same object keeps the same Cryptomatte id across the whole clip.

Pick it in the GUI under **Cryptomatte** (one checkbox runs SAM 3 + the
encoder). CLI: `--passes sam3_matte,cryptomatte`.

## Output (channels.py `CRYPTOMATTE_CHANNELS`)

- `CryptoObject.R/G/B` — colour preview (instances visible at a glance).
- `CryptoObject00/01/02 .R/G/B/A` — 3 ranked levels (6 ranks) of
  `(hash-id, coverage)` pairs.

All **float32** (the ids carry exact MurmurHash3 bit-patterns half-float
would corrupt — the EXR sidecar writer already writes float32). The
manifest (`name -> hex`) and the `cryptomatte/<keyhash>/{name,hash,
conversion,manifest}` header keys are emitted by the pass and stamped onto
every sidecar by the executor, so Nuke's Cryptomatte node reads it natively.

## In Nuke

1. Read the sidecar EXR sequence.
2. Add a **Cryptomatte** node — *Layer Selection* auto-populates
   `CryptoObject` (from the header).
3. Click the woman / a car in the viewer → instant matte. The picker
   resolves the clicked pixel's id float through the manifest.

## Caveat: hard edges

SAM 3 outputs **hard 0/1 masks**, so coverage has no sub-pixel
anti-aliasing — edges are crisp/jaggy (fine for selection, less so for soft
comp). Set `feather` > 0 (Gaussian radius in px) to fake fractional coverage
for softer edges. Default 0 (faithful to SAM 3).

## Params

- `feather` (default 0) — edge-softening radius in pixels.

## Validation

- `tests/test_passes/test_cryptomatte.py`: encoder layout/ranking, the
  spec hash + finite-float guarantee, header keys, a no-model `run_shot`
  manifest round-trip, and a real EXR-writer round-trip (write → read →
  pixel id decodes to the right object).
- Proven end-to-end on a real ACTIONVFX plate: SAM 3 found person + 3
  vehicles, all 4 ids round-tripped at coverage 1.0.
