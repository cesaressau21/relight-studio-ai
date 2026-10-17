"""Application entry point: ``python -m relight_studio [photo-or-project]``."""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path


def setup_logging() -> Path:
    from ..core.paths import logs_dir

    log_file = logs_dir() / "relight_studio.log"
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"),
    ]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=os.environ.get("RELIGHT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )
    for noisy in ("transformers", "huggingface_hub", "PIL", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    log_file = setup_logging()
    log = logging.getLogger("relight_studio")

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from .. import APP_NAME, __version__
    from ..rendering.device import select_device
    from ..ui.main_window import MainWindow
    from ..ui.theme import apply_theme

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName("RelightStudioAI")
    apply_theme(app)

    info = select_device()
    log.info("%s %s starting · device=%s · log=%s", APP_NAME, __version__, info.label, log_file)
    if info.warning:
        log.warning(info.warning)

    window = MainWindow()
    window.show()
    if len(argv) > 1 and Path(argv[1]).exists():
        window.open_path(Path(argv[1]))
    return app.exec()
