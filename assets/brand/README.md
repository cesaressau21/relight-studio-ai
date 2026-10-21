# Relight Studio AI — logo kit

Relight Studio AI is part of the Lumora Studio suite. Its logo follows the Lumora system
(56-unit grid, 4-unit halo gaps, Manrope wordmark, Ink/Paper base) with Relight's own accent.

## The mark

- **Symbol:** a subject (the disc) and a light that orbits it, the way lights are dragged around
  a photo in the app. The orbit is cut by a halo around the light, keeping the 4-unit gap of the
  Lumora mark. The light sits at the upper left, where a key light usually goes.
- **Wordmark:** `RELIGHT` over `STUDIO AI`, both lines the same width; `AI` takes the accent.
- Flat geometry, no gradients. It works in one color and down to 16 px (icons at 32 px and below
  use heavier optics: a thicker orbit and a larger light).

## Files

| Folder | Contents |
| --- | --- |
| `svg/` | Vector masters: `relight-{layout}-{variant}.svg` |
| `png/{layout}/` | Transparent PNG exports: `relight-{layout}-{variant}-{width}.png` |
| `icon/` | App icon and favicon: `relight-app-icon.svg`, `relight-app-icon-small.svg` (≤ 32 px), `favicon.svg`, `relight-app-icon-{16,24,32,48,64,128,256,512}.png`, `relight.ico` (16–256) |
| `source/` | `build.mjs`, which regenerates the whole kit, plus the Manrope fonts and their license |

The build also writes `assets/icon.ico` (executable, installer and shortcuts) and
`assets/icon.png` (window icon). The title bar draws the same mark in code
(`relight_studio/ui/icons.py: logo_pixmap`).

**Layouts:** `horizontal` (primary), `stacked`, `symbol`, `wordmark`.

**Variants:**

| Variant | Use on | Subject / text | Light and orbit | "AI" |
| --- | --- | --- | --- | --- |
| `color-on-dark` | Dark backgrounds | Paper `#ECEBF0` | Orange `#FF9A3C` | `#FF9A3C` |
| `color-on-light` | White or light backgrounds | Ink `#111113` | Orange `#FF9A3C` | `#E07A12` |
| `black` | One-color printing, light backgrounds | Ink `#111113` | Ink | Ink |
| `white` | One-color use on dark or colored backgrounds | White `#FFFFFF` | White | White |

## Colors

| Name | Hex | Role |
| --- | --- | --- |
| Ink | `#111113` | App icon tile, logo on light backgrounds (shared with Lumora Studio) |
| Paper | `#ECEBF0` | Logo on dark backgrounds (shared with Lumora Studio) |
| Relight Orange | `#FF9A3C` | Relight's accent: the light and its orbit, also the app's UI accent |
| Deep Orange | `#E07A12` | `AI` on light backgrounds, where the accent is too pale for small text |

## Typography

[Manrope](https://github.com/googlefonts/manrope) by Mikhail Sharanda: `RELIGHT` in ExtraBold
(800) with +0.2 em tracking, `STUDIO AI` in SemiBold (600) at half the cap height, tracked to
the width of `RELIGHT`. SIL Open Font License 1.1 (`source/fonts/Manrope-OFL.txt`). All text
in the SVG and PNG files is converted to outlines; the font is not needed to use the logo.

## Usage

- **Clear space:** at least the light's diameter (about a fifth of the symbol) around the logo.
- **Minimum size:** symbol at 16 px; horizontal lockup at 160 px wide. Below that, the symbol alone.
- Don't recolor the subject with the accent, add gradients or shadows, rotate, or move the light.

## Rebuilding

```bash
cd assets/brand/source
npm install
npm run build
```
