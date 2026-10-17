"""Reusable form widgets."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QColorDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QWidget,
)

from ..core.color import RGB, rgb_to_hex
from . import theme

_STEPS = 1000


class ResetLabel(QLabel):
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.doubleClicked.emit()


class SliderRow(QWidget):
    """Label + slider + numeric field for a float parameter.

    ``valueChanged`` fires continuously while dragging. Double-clicking the
    label restores the default value.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        default: float = 0.0,
        decimals: int = 2,
        suffix: str = "",
        tooltip: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.minimum, self.maximum, self.default = minimum, maximum, default
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(8)
        self.label = ResetLabel(label)
        self.label.setFixedWidth(86)
        self.label.setToolTip((tooltip + "\n" if tooltip else "") + "Double-click to reset")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, _STEPS)
        self.slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep((maximum - minimum) / 100.0)
        self.spin.setSuffix(suffix)
        self.spin.setFixedWidth(62)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin.setKeyboardTracking(False)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.label.doubleClicked.connect(lambda: self._emit(self.default))
        self.set_value(default)

    def _to_slider(self, value: float) -> int:
        span = self.maximum - self.minimum
        return round((value - self.minimum) / span * _STEPS) if span else 0

    def _from_slider(self, pos: int) -> None:
        value = self.minimum + (self.maximum - self.minimum) * pos / _STEPS
        with QSignalBlocker(self.spin):
            self.spin.setValue(value)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, value: float) -> None:
        with QSignalBlocker(self.slider):
            self.slider.setValue(self._to_slider(value))
        self.valueChanged.emit(value)

    def _emit(self, value: float) -> None:
        self.set_value(value)
        self.valueChanged.emit(self.value())

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, value: float) -> None:
        value = min(max(float(value), self.minimum), self.maximum)
        with QSignalBlocker(self.spin), QSignalBlocker(self.slider):
            self.spin.setValue(value)
            self.slider.setValue(self._to_slider(value))


class ColorButton(QWidget):
    colorChanged = Signal(tuple)

    def __init__(self, color: RGB = (1.0, 1.0, 1.0), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = color
        self.setFixedSize(46, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Pick color")

    def color(self) -> RGB:
        return self._color

    def set_color(self, color: RGB) -> None:
        self._color = tuple(color)  # type: ignore[assignment]
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.setBrush(QColor(rgb_to_hex(self._color)))
        p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 5, 5)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        initial = QColor(rgb_to_hex(self._color))
        dialog = QColorDialog(initial, self)
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dialog.currentColorChanged.connect(lambda c: self._live(c))
        if dialog.exec():
            self._live(dialog.selectedColor())
        else:
            self._live(initial)

    def _live(self, c: QColor) -> None:
        rgb = (c.redF(), c.greenF(), c.blueF())
        if rgb != self._color:
            self.set_color(rgb)
            self.colorChanged.emit(rgb)


def section_title(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("SectionTitle")
    return lbl


def hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("Hint")
    lbl.setWordWrap(True)
    return lbl


def divider() -> QFrame:
    line = QFrame()
    line.setObjectName("Divider")
    return line


def labeled(label: str, widget: QWidget, stretch: bool = True) -> QWidget:
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 1, 0, 1)
    lay.setSpacing(8)
    lbl = QLabel(label)
    lbl.setFixedWidth(86)
    lay.addWidget(lbl)
    lay.addWidget(widget, 1 if stretch else 0)
    if not stretch:
        lay.addStretch(1)
    return row
