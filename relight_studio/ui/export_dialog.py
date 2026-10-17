"""Export options dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.image_io import EXPORT_FORMATS, ExportOptions
from .widgets import ColorButton


class ExportDialog(QDialog):
    def __init__(self, width: int, height: int, default_path: Path, parent: QWidget | None = None,
                 has_alpha: bool = False) -> None:
        super().__init__(parent)
        self.has_alpha = has_alpha
        self.setWindowTitle("Export image")
        self.setMinimumWidth(460)
        self.src_w, self.src_h = width, height
        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setSpacing(8)

        self.format = QComboBox()
        for name in EXPORT_FORMATS:
            self.format.addItem(name, name)
        self.format.currentIndexChanged.connect(self._format_changed)
        form.addRow("Format", self.format)

        self.quality = QSpinBox()
        self.quality.setRange(1, 100)
        self.quality.setValue(95)
        self.quality.setSuffix(" %")
        form.addRow("Quality", self.quality)

        self.size = QComboBox()
        long_side = max(width, height)
        options = [("Original", None)]
        for target in (4096, 3840, 2560, 2048, 1920, 1600, 1080):
            if target < long_side:
                options.append((f"{target} px long side", target))
        for label, value in options:
            self.size.addItem(label, value)
        self.size.currentIndexChanged.connect(self._update_info)
        form.addRow("Resolution", self.size)

        self.transparent = QCheckBox("Keep transparency")
        self.transparent.setChecked(True)
        self.transparent.setToolTip("PNG and WebP keep the image's alpha channel. JPEG has no transparency.")
        self.transparent.toggled.connect(self._format_changed)
        self.background = ColorButton((1.0, 1.0, 1.0))
        self.background.setToolTip("Fill for transparent areas when transparency is not kept")
        bg_row = QHBoxLayout()
        bg_row.addWidget(self.transparent)
        bg_row.addSpacing(12)
        bg_row.addWidget(QLabel("Background"))
        bg_row.addWidget(self.background)
        bg_row.addStretch(1)
        self.transparency_row = QWidget()
        self.transparency_row.setLayout(bg_row)
        self.transparency_row.setVisible(has_alpha)
        form.addRow("Transparency", self.transparency_row)
        form.labelForField(self.transparency_row).setVisible(has_alpha)

        self.icc = QCheckBox("Embed sRGB IEC61966-2.1 profile")
        self.icc.setChecked(True)
        form.addRow("Color profile", self.icc)

        path_row = QHBoxLayout()
        self.path = QLineEdit(str(default_path))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        path_row.addWidget(self.path, 1)
        path_row.addWidget(browse)
        form.addRow("File", path_row)
        lay.addLayout(form)

        self.info = QLabel()
        self.info.setObjectName("Hint")
        lay.addWidget(self.info)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        ok = buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        ok.setObjectName("Accent")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._format_changed()

    def _format_changed(self) -> None:
        fmt = self.format.currentData()
        self.quality.setEnabled(fmt in ("JPEG", "WEBP"))
        if hasattr(self, "transparent"):
            alpha_ok = fmt in ("PNG", "WEBP")
            self.transparent.setEnabled(alpha_ok)
            self.background.setEnabled(not (alpha_ok and self.transparent.isChecked()))
        path = Path(self.path.text())
        self.path.setText(str(path.with_suffix(EXPORT_FORMATS[fmt])))
        self._update_info()

    def _update_info(self) -> None:
        w, h = self.options().target_size(self.src_w, self.src_h)
        self.info.setText(f"Output: {w} × {h} px · rendered at full quality on the export resolution "
                          "(light gizmos are never included).")

    def _browse(self) -> None:
        fmt = self.format.currentData()
        ext = EXPORT_FORMATS[fmt]
        path, _ = QFileDialog.getSaveFileName(self, "Export image", self.path.text(), f"{fmt} (*{ext})")
        if path:
            self.path.setText(str(Path(path).with_suffix(ext)))

    def options(self) -> ExportOptions:
        return ExportOptions(
            format=self.format.currentData(),
            quality=self.quality.value(),
            long_side=self.size.currentData(),
            embed_srgb=self.icc.isChecked(),
            keep_transparency=self.transparent.isChecked(),
            background=self.background.color(),
        )

    def target_path(self) -> Path:
        return Path(self.path.text())
