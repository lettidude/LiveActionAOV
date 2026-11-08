# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""One dark theme for the whole app.

Applied once in `gui.app.main()`: Fusion style + a palette + a short
stylesheet, so the GUI looks the same on Windows and Linux and does not
depend on the OS theme. Widgets opt into the accent colour with
`setObjectName("primary")` (the Run button) - everything else is quiet.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#3d8bff"
_BG = "#1e1f22"  # panels
_BASE = "#161719"  # lists, inputs, viewer surround
_TEXT = "#e4e4e4"
_MUTED = "#9a9a9a"
_BORDER = "#34363b"

STYLESHEET = f"""
QWidget {{ font-size: 9pt; }}
QMainWindow, QDialog {{ background: {_BG}; }}
QStatusBar {{ color: {_MUTED}; border-top: 1px solid {_BORDER}; }}
QStatusBar::item {{ border: none; }}

QPushButton {{
    background: #2a2c31; color: {_TEXT};
    border: 1px solid {_BORDER}; border-radius: 4px;
    padding: 5px 12px; min-height: 18px;
}}
QPushButton:hover {{ background: #33363c; border-color: #45484f; }}
QPushButton:pressed {{ background: #24262a; }}
QPushButton:disabled {{ color: #6a6a6a; background: #232428; border-color: #2c2e33; }}
QPushButton:checked {{ background: #3a3d44; border-color: {ACCENT}; }}

QPushButton#primary {{
    background: {ACCENT}; color: white; font-weight: 600;
    border: 1px solid {ACCENT}; padding: 7px 22px; font-size: 10pt;
}}
QPushButton#primary:hover {{ background: #5399ff; }}
QPushButton#primary:pressed {{ background: #2f78e6; }}
QPushButton#primary:disabled {{ background: #2b3340; color: #7d8794; border-color: #2b3340; }}

QTabWidget::pane {{ border: 1px solid {_BORDER}; border-radius: 4px; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {_MUTED}; padding: 6px 14px;
    border: 1px solid transparent; border-bottom: none;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
}}
QTabBar::tab:selected {{ color: {_TEXT}; background: {_BG}; border-color: {_BORDER}; }}
QTabBar::tab:hover:!selected {{ color: {_TEXT}; }}

QListWidget, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {_BASE}; color: {_TEXT};
    border: 1px solid {_BORDER}; border-radius: 4px; padding: 3px 6px;
    selection-background-color: {ACCENT};
}}
QListWidget::item {{ padding: 4px 2px; }}
QListWidget::item:selected {{ background: #2b3f5e; color: {_TEXT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {_BASE}; border: 1px solid {_BORDER}; }}

QProgressBar {{
    background: {_BASE}; border: 1px solid {_BORDER}; border-radius: 4px;
    text-align: center; color: {_TEXT}; min-height: 20px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QSlider::groove:horizontal {{ height: 4px; background: #34363b; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 12px; margin: -5px 0; border-radius: 6px; background: {_TEXT};
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #3c3f45; min-height: 24px; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: #4a4e55; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #3c3f45; min-width: 24px; border-radius: 5px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

QSplitter::handle {{ background: {_BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QToolTip {{ background: #2a2c31; color: {_TEXT}; border: 1px solid {_BORDER}; padding: 4px; }}
QMenuBar {{ background: {_BG}; color: {_TEXT}; }}
QMenuBar::item:selected {{ background: #2f3238; }}
QMenu {{ background: #26282c; color: {_TEXT}; border: 1px solid {_BORDER}; }}
QMenu::item:selected {{ background: {ACCENT}; }}
QCheckBox, QRadioButton {{ spacing: 6px; }}
"""


def apply_theme(app: QApplication) -> None:
    """Fusion + dark palette + stylesheet. Idempotent."""
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(_BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(_TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(_BASE))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(_BG))
    pal.setColor(QPalette.ColorRole.Text, QColor(_TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor("#2a2c31"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(_TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#2a2c31"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(_TEXT))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#6f7177"))
    pal.setColor(QPalette.ColorRole.Link, QColor(ACCENT))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#6a6a6a"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#6a6a6a"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#6a6a6a"))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)


__all__ = ["ACCENT", "STYLESHEET", "apply_theme"]
