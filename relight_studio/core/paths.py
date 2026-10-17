"""Filesystem locations for bundled resources and user data."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Directory holding ``presets/`` and ``assets/`` (repo root or PyInstaller bundle)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def user_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    path = Path(base) / "RelightStudioAI"
    path.mkdir(parents=True, exist_ok=True)
    return path


def presets_dir() -> Path:
    return resource_root() / "presets"


def assets_dir() -> Path:
    return resource_root() / "assets"


def models_dir() -> Path:
    """Model weight cache. Overridable with ``RELIGHT_MODELS_DIR``."""
    override = os.environ.get("RELIGHT_MODELS_DIR")
    if override:
        path = Path(override)
    elif is_frozen():
        path = user_data_dir() / "models"
    else:
        path = resource_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = user_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
