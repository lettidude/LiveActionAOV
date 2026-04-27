# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""GUI entry point — registered as the `liveaov-gui` console script.

Constructs a QApplication + the three-panel MainWindow and enters the
event loop. Any module-level side-effects (plugin registry discovery,
heavy imports) stay out of import time so the CLI's `--help` — which
imports gui.app transitively via the entry-point manifest — doesn't
pay the Qt startup cost.
"""

from __future__ import annotations

import sys
from importlib.resources import files

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from live_action_aov.gui.main_window import MainWindow

# Unique per-app id used by Windows to group taskbar entries and
# associate a custom icon with the running process. Without this,
# Windows lumps the GUI under the generic Python launcher icon and
# the QApplication.setWindowIcon call only affects the window chrome,
# not the taskbar tile. Format follows the recommended
# CompanyName.ProductName.SubProduct.VersionInformation convention.
_WINDOWS_APP_USER_MODEL_ID = "LeonardoVFX.LiveActionAOV.Gui.1"


def _app_icon_path() -> str:
    """Return the absolute filesystem path to the bundled app icon.

    `importlib.resources.files()` works whether the package is installed
    in a wheel or running editable from `src/`, so the same code path
    serves dev runs and pip installs.
    """
    return str(files("live_action_aov.gui") / "assets" / "app_icon.png")


def _register_windows_taskbar_id() -> None:
    """Tell Windows we're our own app so the taskbar shows our icon.

    Must run BEFORE any QApplication or top-level window is created;
    the shell reads the AppUserModelID at process-attach time. No-op
    on non-Windows platforms.
    """
    if sys.platform != "win32":
        return
    # mypy on Linux narrows `sys.platform != "win32"` to "always true",
    # making the rest of the function unreachable in its analysis. On
    # Windows the narrowing goes the other way and the ignore is
    # unused — `unused-ignore` covers both cases without an OS-specific
    # split.
    try:  # type: ignore[unreachable, unused-ignore]
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_WINDOWS_APP_USER_MODEL_ID)
    except Exception:
        # Failing to set the AppUserModelID just falls back to the
        # generic Python icon — annoying but not fatal. Don't crash
        # the GUI over a cosmetic issue.
        pass


def main() -> int:
    _register_windows_taskbar_id()

    # `QApplication.instance()` is typed as `QCoreApplication | None`,
    # so the `or` expression unions to `QCoreApplication | QApplication`.
    # The runtime guarantee is that we always end up with a
    # `QApplication` (either the existing instance — which is one in
    # this process — or a freshly constructed one). Narrow with an
    # explicit assertion so the `setWindowIcon` call below is
    # statically valid (QCoreApplication doesn't have setWindowIcon).
    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    app.setApplicationName("Live Action AOV")
    app.setOrganizationName("LiveActionAOV")
    # `setWindowIcon` on the QApplication sets the default for every
    # window in this process. MainWindow inherits it; no per-window
    # call needed unless we want a different icon somewhere.
    app.setWindowIcon(QIcon(_app_icon_path()))

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
