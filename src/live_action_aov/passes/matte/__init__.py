"""Matte passes (design §2.5 / spec §13.1 Phase 3).

Shape:
- `sam3.py`  — SAM 3 concept detector + video tracker. Emits semantic
  union masks (`mask.<concept>`) and per-instance hard-mask stacks
  (`sam3_hard_masks`) + a ranked, slotted hero list (`sam3_instances`).
- `rvm.py`   — Robust Video Matting soft-alpha refiner (MIT, commercial OK).
  Reads the SAM 3 artifacts, refines the top-N hard masks into soft alphas,
  packs them into `matte.r/g/b/a`.
- `matanyone2.py` — higher-quality soft-alpha refiner behind the same
  `requires_artifacts` contract as RVM. Non-commercial
  (NTU-S-Lab-1.0). Stamps `utilityPass/matte/commercial = "false"` on
  sidecars so downstream QC can distinguish NC deliverables. Round 2.
- `rank.py`  — pure-Python per-clip hero ranking + slot assignment.
  Consumed by `sam3.py` (to build `sam3_instances`) and by user overrides.
"""
