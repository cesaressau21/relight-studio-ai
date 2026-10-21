"""Release hygiene: the application's license and the third-party notices shipped with the build."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packaging"))

from collect_licenses import INDEX, collect  # noqa: E402


def test_project_is_mit_licensed():
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("MIT License") and "Copyright (c)" in text
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"]["text"].startswith("MIT")


def test_bundled_packages_ship_their_licenses(tmp_path):
    rows = collect(tmp_path)
    names = {name.lower() for name, _, _ in rows}
    assert {"pyside6", "shiboken6", "torch", "transformers", "numpy", "pillow"} <= names
    assert not {"pytest", "pyinstaller"} & names  # build and test tools are not shipped
    assert any((tmp_path / "torch").rglob("LICENSE*"))
    index = (tmp_path / INDEX).read_text(encoding="utf-8")
    assert "PySide6" in index and "LGPL-3.0" in index


def test_qt_lgpl_texts_are_shipped(tmp_path):
    collect(tmp_path)
    lgpl = (tmp_path / "LGPL-3.0.txt").read_text(encoding="utf-8")
    gpl = (tmp_path / "GPL-3.0.txt").read_text(encoding="utf-8")
    assert "GNU LESSER GENERAL PUBLIC LICENSE" in lgpl and "Version 3, 29 June 2007" in lgpl
    assert "GNU GENERAL PUBLIC LICENSE" in gpl and "Version 3, 29 June 2007" in gpl


def test_build_spec_copies_the_licenses():
    spec = (ROOT / "packaging" / "relight_studio.spec").read_text(encoding="utf-8")
    assert '"LICENSE"' in spec and "LICENSES.md" in spec and "collect(" in spec


def test_version_is_the_same_everywhere():
    import re

    import relight_studio

    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert relight_studio.__version__ == version
    iss = (ROOT / "packaging" / "installer" / "relight_studio.iss").read_text(encoding="utf-8")
    assert re.search(r'#define AppVersion "([^"]+)"', iss).group(1) == version
    assert (ROOT / "docs" / "releases" / f"v{version}-beta.md").exists()  # every version ships notes
