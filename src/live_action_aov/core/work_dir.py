# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Where passes put their temporary files (frames exported from the plate).

Default: the system temp folder. Studios with content-security rules point
`LAOV_WORK_ROOT` (CLI `--work-root`, GUI File > Temp folder) at a LOCAL disk
inside the project tree, so plate-derived frames never sit in a user's
%TEMP%. Keep it local: COLMAP's database is SQLite, which is unreliable on
a network share. Child processes inherit the variable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ENV_VAR = "LAOV_WORK_ROOT"


def work_root() -> Path | None:
    """The configured root, or None for the system temp folder."""
    v = os.environ.get(ENV_VAR, "").strip()
    return Path(v) if v else None


def set_work_root(path: str | Path | None) -> None:
    """Set (or clear, with None / "") the root for this process and its children."""
    if path is None or not str(path).strip():
        os.environ.pop(ENV_VAR, None)
    else:
        os.environ[ENV_VAR] = str(path).strip()


def make_work_dir(prefix: str) -> Path:
    """A fresh, empty folder under the work root. The caller removes it."""
    root = work_root()
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


__all__ = ["ENV_VAR", "make_work_dir", "set_work_root", "work_root"]
