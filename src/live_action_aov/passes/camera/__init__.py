"""Camera passes — classical SfM (pycolmap) in v2a; learned backends in v2b+.

The `CameraPassStub` from v1 remains so tests / entry-point scans that
referenced it keep working. Real backends live next to it:

- `PyColmapCameraPass` (v2a, BSD, commercial-safe) — classical incremental
  SfM via the pycolmap PyPI wheels.
- `MapAnythingCameraPass` (v2b, experimental) — learned reconstructor.
- `MegaSAMCameraPass` (v2c, nuclear option) — subprocess-isolated because
  of CUDA-extension build complexity.
"""
