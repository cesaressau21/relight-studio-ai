"""Build the lightweight Windows installer (dist/installer/RelightStudioAI-<version>-beta-setup.exe).

    .venv/Scripts/python.exe packaging/build_installer.py

Needs Inno Setup 6 (``winget install JRSoftware.InnoSetup``); ``ISCC`` is looked up on PATH
and in the default per-user and per-machine install folders, or set ``ISCC`` to its path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "packaging" / "installer" / "relight_studio.iss"


def find_iscc() -> Path | None:
    candidates = [os.environ.get("ISCC"), shutil.which("ISCC")]
    for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if base:
            candidates.append(str(Path(base) / ("Programs" if base == os.environ.get("LOCALAPPDATA") else "")
                                  / "Inno Setup 6" / "ISCC.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def build() -> Path:
    iscc = find_iscc()
    if iscc is None:
        raise SystemExit("Inno Setup 6 was not found: winget install JRSoftware.InnoSetup")
    ver = version()
    subprocess.run([str(iscc), "/Q", f"/DAppVersion={ver}", str(SCRIPT)], check=True)
    return ROOT / "dist" / "installer" / f"RelightStudioAI-{ver}-beta-setup.exe"


if __name__ == "__main__":
    out = build()
    print(f"{out} ({out.stat().st_size / 2**20:.1f} MB)")
    sys.exit(0)
