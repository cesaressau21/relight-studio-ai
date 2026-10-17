"""Source files are valid UTF-8 without double-encoded (mojibake) text."""

from __future__ import annotations

import json
import re

import pytest

from .conftest import ROOT

FILES = sorted(
    p for pattern in ("relight_studio/**/*.py", "tests/*.py", "presets/*.json", "docs/*.md", "README.md")
    for p in ROOT.glob(pattern)
)
# UTF-8 bytes of characters like "×", "•", "…" wrongly decoded as cp1252: a lead byte
# rendered as U+00C2/U+00C3/U+00E2 followed by a continuation-byte character.
MOJIBAKE = re.compile("[ÂÃâ][\u0080-¿Œ-™]")


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_file_is_clean_utf8(path):
    text = path.read_bytes().decode("utf-8")
    match = MOJIBAKE.search(text)
    assert match is None, f"double-encoded text near: {text[max(0, match.start() - 20): match.end() + 20]!r}"


@pytest.mark.parametrize("path", sorted(ROOT.glob("presets/*.json")), ids=lambda p: p.name)
def test_presets_are_valid_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] and data["lights"]
