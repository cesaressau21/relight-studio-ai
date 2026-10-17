"""Screen-fitting helpers: initial window placement and widgets that must not force width."""

from __future__ import annotations

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QComboBox, QLabel, QSizePolicy, QWidget

PREFERRED_SIZE = (1600, 1000)
MIN_COMFORTABLE = (1300, 760)  # below this the editor opens maximized
TITLE_BAR = 32


def initial_window_rect(available: QRect) -> tuple[QRect, bool]:
    """Window geometry (client area) centered in ``available``, and whether to maximize.

    Small screens (for example 1920x1080 at 150 % scaling = 1280x672 logical)
    get a maximized window so every panel has room.
    """
    usable_h = available.height() - TITLE_BAR
    if available.width() < MIN_COMFORTABLE[0] or usable_h < MIN_COMFORTABLE[1]:
        return QRect(available.x(), available.y() + TITLE_BAR, available.width(), usable_h), True
    w = min(PREFERRED_SIZE[0], int(available.width() * 0.92))
    h = min(PREFERRED_SIZE[1], int(usable_h * 0.94))
    x = available.x() + (available.width() - w) // 2
    y = available.y() + TITLE_BAR + (usable_h - h) // 2
    return QRect(x, y, w, h), False


def compact_combo(combo: QComboBox, chars: int = 12) -> QComboBox:
    """Let a combo box shrink with its panel instead of sizing to its longest item."""
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)
    combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    # Some item sets still report a wide size hint; an explicit minimum wins in layouts.
    combo.setMinimumWidth(min(combo.minimumSizeHint().width(), 140))
    return combo


def compact_all_combos(root: QWidget) -> None:
    for combo in root.findChildren(QComboBox):
        compact_combo(combo)


def elastic_label(label: QLabel) -> QLabel:
    """A single-line label that may be cut instead of widening its window."""
    label.setMinimumWidth(40)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return label
