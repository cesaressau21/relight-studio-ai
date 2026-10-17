"""Dark theme: palette tokens and the application stylesheet."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BG = "#111113"
SURFACE = "#18181b"
RAISED = "#202024"
HOVER = "#2a2a2f"
BORDER = "#2b2b31"
TEXT = "#e8e8ec"
MUTED = "#8d8d97"
FAINT = "#5d5d66"
ACCENT = "#ff9a3c"
ACCENT_SOFT = "#3a2a1c"
DANGER = "#ff5d5d"

FONT_FAMILY = "Segoe UI Variable Text"


STYLESHEET = f"""
* {{ outline: 0; }}
QWidget {{ background: {BG}; color: {TEXT}; font-size: 9pt; }}
QWidget#Panel, QWidget#TopBar, QWidget#ToolRail, QStatusBar {{ background: {SURFACE}; }}
QWidget#TopBar {{ border-bottom: 1px solid {BORDER}; }}
QWidget#ToolRail {{ border-right: 1px solid {BORDER}; }}
QWidget#Panel {{ border-left: 1px solid {BORDER}; }}
QStatusBar {{ border-top: 1px solid {BORDER}; color: {MUTED}; }}
QStatusBar QLabel {{ background: transparent; color: {MUTED}; padding: 0 8px; }}
QLabel {{ background: transparent; }}
QLabel#Brand {{ font-size: 11pt; font-weight: 600; letter-spacing: 0.3px; }}
QLabel#BrandAccent {{ font-size: 11pt; font-weight: 600; color: {ACCENT}; }}
QLabel#SectionTitle {{ color: {MUTED}; font-size: 8pt; font-weight: 600; letter-spacing: 1px; padding-top: 6px; }}
QLabel#Hint {{ color: {FAINT}; font-size: 8pt; }}
QLabel#Value {{ color: {MUTED}; }}

QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 7px; padding: 5px; color: {TEXT}; }}
QToolButton:hover {{ background: {HOVER}; }}
QToolButton:checked {{ background: {ACCENT_SOFT}; border-color: #5a3b20; color: {ACCENT}; }}
QToolButton:disabled {{ color: {FAINT}; }}
QToolButton#TopAction {{ padding: 5px 10px; }}
QToolButton#Primary {{ background: {ACCENT}; color: #1a1006; font-weight: 600; padding: 5px 12px; }}
QToolButton#Primary:hover {{ background: #ffad5e; }}
QToolButton#Primary:disabled {{ background: {RAISED}; color: {FAINT}; }}

QPushButton {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 10px; }}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {FAINT}; }}
QPushButton#Accent {{ background: {ACCENT}; border-color: {ACCENT}; color: #1a1006; font-weight: 600; }}

QTabWidget::pane {{ border: none; background: {SURFACE}; }}
QTabBar {{ background: {SURFACE}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 10px; border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

QScrollArea {{ border: none; background: {SURFACE}; }}
QScrollArea > QWidget > QWidget {{ background: {SURFACE}; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ height: 0; }}

QSlider::groove:horizontal {{ height: 3px; background: {BORDER}; border-radius: 1px; }}
QSlider::sub-page:horizontal {{ background: #6b5440; border-radius: 1px; }}
QSlider::handle:horizontal {{ background: {TEXT}; width: 11px; height: 11px; margin: -4px 0; border-radius: 5px; }}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QSlider::handle:horizontal:disabled {{ background: {FAINT}; }}

QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 5px; padding: 2px 4px; selection-background-color: {ACCENT};
}}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus, QLineEdit:focus {{ border-color: #6b5440; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button, QSpinBox::up-button, QSpinBox::down-button {{ width: 0; }}
QComboBox::drop-down {{ border: none; width: 16px; }}
QComboBox QAbstractItemView {{ background: {RAISED}; border: 1px solid {BORDER}; selection-background-color: {HOVER}; }}

QCheckBox {{ spacing: 6px; background: transparent; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 4px; border: 1px solid {FAINT}; background: {RAISED}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QListWidget {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 2px; }}
QListWidget::item {{ border-radius: 5px; padding: 0; }}
QListWidget::item:selected {{ background: {HOVER}; color: {TEXT}; }}
QListWidget::item:hover {{ background: {RAISED}; }}

QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px 6px 10px; border-radius: 4px; }}
QMenu::item:selected {{ background: {HOVER}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 6px; }}

QProgressBar {{ background: {RAISED}; border: none; border-radius: 3px; height: 6px; max-height: 6px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}
QDialog {{ background: {SURFACE}; }}
QFrame#Divider {{ background: {BORDER}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#Card {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#Card:hover {{ border-color: #4a3a2c; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    font = QFont(FONT_FAMILY)
    if not font.exactMatch():
        font = QFont("Segoe UI")
    font.setPointSizeF(9.0)
    app.setFont(font)
    pal = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, BG),
        (QPalette.ColorRole.Base, RAISED),
        (QPalette.ColorRole.AlternateBase, SURFACE),
        (QPalette.ColorRole.Text, TEXT),
        (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.Button, RAISED),
        (QPalette.ColorRole.ButtonText, TEXT),
        (QPalette.ColorRole.Highlight, ACCENT),
        (QPalette.ColorRole.HighlightedText, "#1a1006"),
        (QPalette.ColorRole.ToolTipBase, RAISED),
        (QPalette.ColorRole.ToolTipText, TEXT),
        (QPalette.ColorRole.PlaceholderText, FAINT),
    ):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)
