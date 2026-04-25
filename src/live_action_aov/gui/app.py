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

from PySide6.QtWidgets import QApplication

from live_action_aov.gui.main_window import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Live Action AOV")
    app.setOrganizationName("LiveActionAOV")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
