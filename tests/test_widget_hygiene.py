"""Widgets must not hide Qt methods behind attributes (``self.size = QComboBox()`` breaks ``size()``)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import SignalInstance
from PySide6.QtWidgets import QWidget

from relight_studio.ui.export_dialog import ExportDialog

def qt_methods(widget: QWidget) -> set[str]:
    """Public methods of the Qt class the widget derives from."""
    base = next(c for c in type(widget).__mro__ if c.__module__.startswith("PySide6"))
    return {name for name in dir(base) if not name.startswith("_") and callable(getattr(base, name, None))}


def shadowed(widget: QWidget) -> list[str]:
    found = []
    for child in [widget, *widget.findChildren(QWidget)]:
        methods = qt_methods(child)
        for name, value in getattr(child, "__dict__", {}).items():
            if name in methods and not isinstance(value, SignalInstance):  # PySide caches bound signals
                found.append(f"{type(child).__name__}.{name}")
    return found


def test_editor_widgets_keep_qt_methods(qapp, editor):
    assert shadowed(editor) == []


def test_export_dialog_keeps_qt_methods(qapp, tmp_path):
    dialog = ExportDialog(1500, 1000, Path(tmp_path) / "out.png", has_alpha=True)
    assert shadowed(dialog) == []
    assert dialog.size().width() > 0
