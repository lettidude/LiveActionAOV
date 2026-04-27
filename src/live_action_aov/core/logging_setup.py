# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Run-scoped file logging for the pipeline.

Why this module exists
----------------------
Before this module, the only sink for `logging` records was the GUI's
`LogPanel` (see `gui/log_panel.py`), which lives in memory and is gone
as soon as the GUI is closed. The CLI had no logger at all. That made
post-mortem ("what happened during yesterday's run?") impossible.

What we write
-------------
Each `RunLoggingSession` opens three handlers and closes them on exit:

1. **Per-run log next to the sidecars** —
     `<sidecar_dir>/submit_<YYYYMMDD-HHMMSS>.log`
   Travels with the shot. Archival-grade: anyone touching the delivery
   folder can audit the run without the tool.

2. **Central mirror** —
     `<user_log_dir>/submit_<YYYYMMDD-HHMMSS>_<shot>.log`
   One place to answer "what did I run last week?" across many
   projects. `user_log_dir` resolves via `platformdirs`, so on
   Windows it lands in `%LOCALAPPDATA%\\LiveActionAOV\\Logs`.

3. **Rolling warnings/errors digest** —
     `<user_log_dir>/warnings.log`
   WARNING+ records only, appended forever (rotated at 1 MB × 3
   backups). The one file to skim when something went wrong but you
   don't remember which run.

The central dir is pruned to the last 30 per-run files on each open
so a year of renders doesn't balloon `%LOCALAPPDATA%`.

All three handlers attach to the **root** logger, so every
`logging.getLogger(...)` in the project (plus diffusers / transformers
/ huggingface_hub / our own code) lands in the files for free. The
GUI's Qt handler stays attached independently — the live panel keeps
working regardless of whether a run session is open.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal

from platformdirs import user_log_dir

# Passing `appauthor=False` to platformdirs stops the Windows path
# from getting an extra `<APPDATA>\LiveActionAOV\LiveActionAOV\Logs`
# double-nest; the result is the cleaner `<APPDATA>\LiveActionAOV\Logs`.
# Type is `str | Literal[False]` (not `str | bool`) because platformdirs
# accepts the False sentinel specifically, not arbitrary bools.
_APP_NAME = "LiveActionAOV"
_APP_AUTHOR: str | Literal[False] = False

_MAX_CENTRAL_LOGS = 30
_WARNINGS_FILENAME = "warnings.log"
_WARNINGS_MAX_BYTES = 1_000_000
_WARNINGS_BACKUP_COUNT = 3

_FILE_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_log_dir() -> Path:
    """Return the central log directory, creating it if needed.

    GUI menu items, CLI `--log-dir` introspection, and docs all resolve
    to the same path via this function.
    """
    path = Path(user_log_dir(_APP_NAME, _APP_AUTHOR))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(name: str) -> str:
    """Turn a shot name into something safe for a filename on every OS."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return slug or "shot"


@dataclass(frozen=True)
class RunLogPaths:
    """Paths written by a `RunLoggingSession` — useful for error dialogs
    and for stamping into sidecar metadata."""

    per_run: Path
    central: Path
    warnings_digest: Path


class RunLoggingSession:
    """Context manager that attaches file handlers to the root logger
    for the duration of one submit.

    Usage:

        with RunLoggingSession(shot_name="sh020", sidecar_dir=out) as paths:
            ...  # run the pipeline; every `logging.*` call lands in `paths`

    Handlers are removed and flushed on exit, including on exception —
    the traceback is logged as ERROR before the context unwinds.
    """

    def __init__(self, shot_name: str, sidecar_dir: Path) -> None:
        self._shot_name = shot_name
        self._sidecar_dir = sidecar_dir
        self._timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self._handlers: list[logging.Handler] = []
        self.paths: RunLogPaths | None = None

    def __enter__(self) -> RunLogPaths:
        sidecar_dir = self._sidecar_dir
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        central_dir = get_log_dir()

        # Prune before opening so a crashed previous session doesn't leave
        # an orphaned file undiscoverable behind the retention window.
        _prune_central_logs(central_dir, keep=_MAX_CENTRAL_LOGS)

        per_run = sidecar_dir / f"submit_{self._timestamp}.log"
        central = central_dir / f"submit_{self._timestamp}_{_slugify(self._shot_name)}.log"
        warnings_digest = central_dir / _WARNINGS_FILENAME

        formatter = logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT)

        per_run_h = logging.FileHandler(per_run, mode="w", encoding="utf-8")
        per_run_h.setLevel(logging.INFO)
        per_run_h.setFormatter(formatter)

        central_h = logging.FileHandler(central, mode="w", encoding="utf-8")
        central_h.setLevel(logging.INFO)
        central_h.setFormatter(formatter)

        warnings_h = logging.handlers.RotatingFileHandler(
            warnings_digest,
            maxBytes=_WARNINGS_MAX_BYTES,
            backupCount=_WARNINGS_BACKUP_COUNT,
            encoding="utf-8",
        )
        warnings_h.setLevel(logging.WARNING)
        warnings_h.setFormatter(formatter)

        root = logging.getLogger()
        # Root logger defaults to WARNING — we need INFO to reach the
        # file handlers. Don't lower below INFO: DEBUG from tokenisers
        # and diffusers floods gigabytes of noise that hides real events.
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)

        for h in (per_run_h, central_h, warnings_h):
            root.addHandler(h)
            self._handlers.append(h)

        self.paths = RunLogPaths(
            per_run=per_run,
            central=central,
            warnings_digest=warnings_digest,
        )

        header = (
            f"===== Submit start: {self._shot_name} =====\n"
            f"  timestamp  : {self._timestamp}\n"
            f"  sidecar_dir: {sidecar_dir}\n"
            f"  python     : {sys.version.split()[0]}\n"
            f"  platform   : {sys.platform}"
        )
        logging.getLogger("live_action_aov").info(header)
        return self.paths

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        log = logging.getLogger("live_action_aov")
        if exc_type is not None:
            # Capture the traceback into the files before the handlers
            # are torn down — otherwise the caller's except-block sees
            # the exception and the post-mortem has no record of why.
            tb_text = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            log.error(
                "===== Submit FAILED: %s =====\n%s",
                self._shot_name,
                tb_text.rstrip(),
            )
        else:
            log.info("===== Submit done: %s =====", self._shot_name)

        root = logging.getLogger()
        for h in self._handlers:
            try:
                h.flush()
            except Exception:
                pass
            try:
                root.removeHandler(h)
            except Exception:
                pass
            try:
                h.close()
            except Exception:
                pass
        self._handlers.clear()


def _prune_central_logs(central_dir: Path, *, keep: int) -> None:
    """Delete the oldest per-run files until at most `keep` remain.

    `warnings.log*` (the rotating digest) is always preserved.
    """
    try:
        files = sorted(
            (p for p in central_dir.glob("submit_*.log") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in files[keep:]:
        try:
            stale.unlink()
        except OSError:
            pass


__all__ = [
    "RunLogPaths",
    "RunLoggingSession",
    "get_log_dir",
]
