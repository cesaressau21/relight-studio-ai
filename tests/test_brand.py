"""The logo kit and the places the app uses it."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtGui import QColor

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "assets" / "brand"
LAYOUTS = ("horizontal", "stacked", "symbol", "wordmark")
VARIANTS = ("color-on-light", "color-on-dark", "black", "white")


def test_kit_has_every_layout_and_variant_as_flat_outlined_svg():
    for layout in LAYOUTS:
        for variant in VARIANTS:
            svg = (BRAND / "svg" / f"relight-{layout}-{variant}.svg").read_text(encoding="utf-8")
            assert "Gradient" not in svg and "<text" not in svg, (layout, variant)  # flat, no font needed
            assert (BRAND / "png" / layout).glob(f"relight-{layout}-{variant}-*.png")


def test_app_icons_cover_windows_sizes():
    with Image.open(ROOT / "assets" / "icon.ico") as ico:
        sizes = {s[0] for s in ico.info["sizes"]}
    assert {16, 24, 32, 48, 64, 128, 256} <= sizes
    with Image.open(ROOT / "assets" / "icon.png") as png:
        assert png.size == (256, 256)


def test_title_bar_mark_is_the_flat_orbit(qapp):
    from relight_studio.ui import icons, theme

    image = icons.logo_pixmap(56).toImage()  # 2x device ratio: 112 px, 2 px per grid unit
    def at(x, y):
        return image.pixelColor(int(x * 2), int(y * 2))
    def near(c: QColor, ref: str) -> bool:
        r = QColor(ref)
        return abs(c.red() - r.red()) + abs(c.green() - r.green()) + abs(c.blue() - r.blue()) < 30

    assert near(at(28, 28), theme.TEXT)            # the subject, flat
    assert near(at(12.4, 12.4), theme.ACCENT)       # the light, upper left
    assert near(at(50, 28), theme.ACCENT)           # the orbit
    assert at(28, 2).alpha() == 0 or not near(at(28, 2), theme.TEXT)
