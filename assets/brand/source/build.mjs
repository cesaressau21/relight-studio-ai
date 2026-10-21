// Builds the Relight Studio AI logo kit: SVG sources, PNG exports, app icon and ICO.
// Run from this folder: npm install && npm run build
// The kit follows the Lumora Studio system (56-unit grid, 4-unit halo gaps, Manrope wordmark,
// Ink/Paper base) with Relight's own accent.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as opentypeNs from 'opentype.js';
import { Resvg } from '@resvg/resvg-js';
import pngToIco from 'png-to-ico';

const opentype = opentypeNs.parse ? opentypeNs : opentypeNs.default;
const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(here, '..');
const appAssets = path.resolve(here, '..', '..');

export const COLORS = {
  ink: '#111113',
  paper: '#ECEBF0',
  white: '#FFFFFF',
  accent: '#FF9A3C', // Relight orange: the light and its orbit
  accentText: '#E07A12', // "AI" on light backgrounds, where #FF9A3C is too pale for small text
};

// Symbol on a 56-unit grid: the subject (a disc) and a light that orbits it, the way lights
// are dragged around a photo in the app. The orbit is cut by a halo around the light, keeping
// the 4-unit gap of the Lumora mark. Small icons (32 px and below) use heavier optics.
const S = 56;
const C = 28;
const LIGHT_ANGLE = (225 * Math.PI) / 180; // upper left, where key lights usually sit
const OPTICS = {
  regular: { orbit: 23, stroke: 3.6, subject: 14.5, light: 6, halo: 4 },
  small: { orbit: 22, stroke: 5.4, subject: 13.5, light: 7.2, halo: 3.6 },
};

const r2 = (n) => Math.round(n * 1000) / 1000;
let uid = 0;

function symbolGroup(x, y, size, fg, light, optics = 'regular') {
  const o = OPTICS[optics];
  const k = size / S;
  const lx = C + o.orbit * Math.cos(LIGHT_ANGLE);
  const ly = C + o.orbit * Math.sin(LIGHT_ANGLE);
  const id = `relight-halo-${uid++}`;
  return `<g transform="translate(${r2(x)} ${r2(y)}) scale(${r2(k)})">`
    + `<mask id="${id}" maskUnits="userSpaceOnUse" x="0" y="0" width="${S}" height="${S}">`
    + `<rect width="${S}" height="${S}" fill="#fff"/>`
    + `<circle cx="${r2(lx)}" cy="${r2(ly)}" r="${r2(o.light + o.halo)}" fill="#000"/></mask>`
    + `<circle cx="${C}" cy="${C}" r="${o.orbit}" fill="none" stroke="${light}" stroke-width="${o.stroke}" mask="url(#${id})"/>`
    + `<circle fill="${fg}" cx="${C}" cy="${C}" r="${o.subject}"/>`
    + `<circle fill="${light}" cx="${r2(lx)}" cy="${r2(ly)}" r="${o.light}"/></g>`;
}

function loadFont(file) {
  const buf = fs.readFileSync(path.join(here, 'fonts', file));
  return opentype.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

const bold = loadFont('Manrope-ExtraBold.ttf');
const medium = loadFont('Manrope-SemiBold.ttf');

// Lays out a line with letter tracking (in em) so that its cap height equals `cap`.
function layoutLine(font, text, cap, trackingEm, baseline) {
  const size = (cap * font.unitsPerEm) / font.tables.os2.sCapHeight;
  const unit = size / font.unitsPerEm;
  const glyphs = font.stringToGlyphs(text);
  let x = 0;
  const placed = glyphs.map((g, i) => {
    if (i > 0) x += font.getKerningValue(glyphs[i - 1], g) * unit + trackingEm * size;
    const at = x;
    x += g.advanceWidth * unit;
    return { g, at };
  });
  const box = { x1: Infinity, y1: Infinity, x2: -Infinity, y2: -Infinity };
  for (const p of placed) {
    const b = p.g.getPath(p.at, baseline, size).getBoundingBox();
    if (!Number.isFinite(b.x1)) continue; // the space
    box.x1 = Math.min(box.x1, b.x1); box.y1 = Math.min(box.y1, b.y1);
    box.x2 = Math.max(box.x2, b.x2); box.y2 = Math.max(box.y2, b.y2);
  }
  return { size, placed, baseline, box };
}

function glyphPath(line, dx, from = 0, to = Infinity) {
  return line.placed.slice(from, to)
    .map((p) => p.g.getPath(p.at + dx, line.baseline, line.size).toPathData(3)).join('');
}

// Wordmark: RELIGHT (Manrope ExtraBold, +0.2 em) over STUDIO AI (Manrope SemiBold, half the cap
// height), tracked so both lines share the same ink width. "AI" takes the accent color.
function buildWordmark(cap) {
  const capSmall = cap * 0.5;
  const gap = cap * 0.45;
  const baseline = cap + gap + capSmall;
  const top = layoutLine(bold, 'RELIGHT', cap, 0.2, cap);
  const width = top.box.x2 - top.box.x1;
  // STUDIO and AI share one tracking; the space between them is twice the letter gap, not a
  // tracked space glyph (which would read as a hole). Solve the tracking for the full width.
  const lineWidth = (t) => {
    const a = layoutLine(medium, 'STUDIO', capSmall, t, baseline);
    const b = layoutLine(medium, 'AI', capSmall, t, baseline);
    const space = 2 * t * a.size + 0.12 * a.size;
    return { a, b, space, w: (a.box.x2 - a.box.x1) + space + (b.box.x2 - b.box.x1) };
  };
  let lo = 0, hi = 2;
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) / 2;
    if (lineWidth(mid).w < width) lo = mid; else hi = mid;
  }
  const { a, b, space } = lineWidth(lo);
  const bx = (a.box.x2 - a.box.x1) + space - b.box.x1;
  return {
    main: glyphPath(top, -top.box.x1) + glyphPath(a, -a.box.x1),
    ai: glyphPath(b, bx),
    width,
    height: cap + gap + capSmall,
    inkTop: top.box.y1,
    inkBottom: Math.max(top.box.y2, a.box.y2, b.box.y2),
  };
}

const CAP = 20.7; // same proportions as the Lumora Studio lockup
const word = buildWordmark(CAP);

function svgDoc(minX, minY, w, h, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${r2(minX)} ${r2(minY)} ${r2(w)} ${r2(h)}" role="img" aria-label="Relight Studio AI">`
    + `<title>Relight Studio AI</title>${body}</svg>\n`;
}

function wordGroup(x, y, fill, accent) {
  return `<g transform="translate(${r2(x)} ${r2(y)})"><path fill="${fill}" d="${word.main}"/>`
    + `<path fill="${accent}" d="${word.ai}"/></g>`;
}

const LAYOUTS = {
  horizontal(c) {
    const gap = S * 0.34;
    const ty = (S - word.height) / 2;
    const minY = Math.min(0, ty + word.inkTop);
    const maxY = Math.max(S, ty + word.inkBottom);
    return svgDoc(0, minY, S + gap + word.width, maxY - minY,
      symbolGroup(0, 0, S, c.fg, c.light) + wordGroup(S + gap, ty, c.text, c.ai));
  },
  stacked(c) {
    const gap = S * 0.36;
    const w = Math.max(S, word.width);
    const ty = S + gap;
    return svgDoc(0, 0, w, ty + word.inkBottom,
      symbolGroup((w - S) / 2, 0, S, c.fg, c.light) + wordGroup((w - word.width) / 2, ty, c.text, c.ai));
  },
  symbol(c) {
    return svgDoc(0, 0, S, S, symbolGroup(0, 0, S, c.fg, c.light));
  },
  wordmark(c) {
    return svgDoc(0, word.inkTop, word.width, word.inkBottom - word.inkTop, wordGroup(0, 0, c.text, c.ai));
  },
};

const VARIANTS = {
  'color-on-light': { fg: COLORS.ink, light: COLORS.accent, text: COLORS.ink, ai: COLORS.accentText },
  'color-on-dark': { fg: COLORS.paper, light: COLORS.accent, text: COLORS.paper, ai: COLORS.accent },
  black: { fg: COLORS.ink, light: COLORS.ink, text: COLORS.ink, ai: COLORS.ink },
  white: { fg: COLORS.white, light: COLORS.white, text: COLORS.white, ai: COLORS.white },
};

const PNG_WIDTHS = {
  horizontal: [600, 1200, 2400],
  stacked: [400, 800, 1600],
  symbol: [64, 128, 256, 512, 1024],
  wordmark: [600, 1200, 2400],
};

function write(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, data);
  return file;
}

function renderPng(svg, width) {
  return new Resvg(svg, { fitTo: { mode: 'width', value: width } }).render().asPng();
}

const written = [];
for (const [layout, make] of Object.entries(LAYOUTS)) {
  for (const [variant, colors] of Object.entries(VARIANTS)) {
    const name = `relight-${layout}-${variant}`;
    const svg = make(colors);
    written.push(write(path.join(outDir, 'svg', `${name}.svg`), svg));
    for (const w of PNG_WIDTHS[layout]) {
      written.push(write(path.join(outDir, 'png', layout, `${name}-${w}.png`), renderPng(svg, w)));
    }
  }
}

// App icon: the symbol on an Ink rounded tile, like the Lumora Studio app icon.
function appIcon(optics) {
  const size = 512;
  const inner = size * 0.64;
  const o = (size - inner) / 2;
  return svgDoc(0, 0, size, size,
    `<rect width="${size}" height="${size}" rx="${size * 0.22}" fill="${COLORS.ink}"/>`
    + symbolGroup(o, o, inner, COLORS.paper, COLORS.accent, optics));
}

const icon = appIcon('regular');
const iconSmall = appIcon('small');
written.push(write(path.join(outDir, 'icon', 'relight-app-icon.svg'), icon));
written.push(write(path.join(outDir, 'icon', 'relight-app-icon-small.svg'), iconSmall));
written.push(write(path.join(outDir, 'icon', 'favicon.svg'), iconSmall));
const ICON_SIZES = [16, 24, 32, 48, 64, 128, 256, 512];
const iconPngs = {};
for (const s of ICON_SIZES) {
  iconPngs[s] = renderPng(s <= 32 ? iconSmall : icon, s);
  written.push(write(path.join(outDir, 'icon', `relight-app-icon-${s}.png`), iconPngs[s]));
}
const ico = await pngToIco([16, 24, 32, 48, 64, 128, 256].map((s) => iconPngs[s]));
written.push(write(path.join(outDir, 'icon', 'relight.ico'), ico));

// The files the application and its installer use.
written.push(write(path.join(appAssets, 'icon.ico'), ico));
written.push(write(path.join(appAssets, 'icon.png'), iconPngs[256]));

console.log(`Wrote ${written.length} files`);
console.log(`Wordmark: ${r2(word.width)} x ${r2(word.height)} units (symbol ${S})`);
