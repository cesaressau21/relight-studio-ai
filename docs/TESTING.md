# Test suite

Run everything (≈ 35 s on an RTX 5070):

```bash
.venv/Scripts/python.exe -m pytest
```

Fast run without the real AI models:

```bash
.venv/Scripts/python.exe -m pytest -m "not integration"
```

UI tests run Qt offscreen (`QT_QPA_PLATFORM=offscreen`, set in `tests/conftest.py`) and keep
their settings in a temporary INI file, so they never touch your real configuration. The
`editor` fixture opens a synthetic photo with injected depth, so UI tests need no model.

| Area | File | What is protected |
|---|---|---|
| Engine physics | `test_engine.py` | identity with no lights, geometry-dependent shading, cast shadows, rim light, mask, chunked export = preview, view modes |
| Every engine parameter | `test_engine_params.py` | depth remap/invert/gamma/relief/smoothing/offset, each light property (softness, diffusion, falloff, Kelvin, specular, roughness, area size, direction), lighting model settings, each adjustment, filters |
| Silhouette edges | `test_edges.py` | trimap from depth, alpha refinement, no halo on the background next to a subject, layered depth |
| Document model | `test_core.py` | colors, lights, history merge/limit, brushes, presets |
| Files | `test_io.py` | EXIF/alpha loading, PNG/JPEG/WebP export + ICC, project round trip, version checks |
| Editor logic | `test_controller.py` | every edit command, undo/redo per kind, strokes, presets, dirty flag |
| Panels | `test_ui_panels.py` | every slider/checkbox/combo writes the document and refreshes after undo |
| Canvas | `test_canvas.py`, `test_ui_canvas_interaction.py` | real mouse/wheel/keyboard: drag lights, place tools, modifiers, brushes, pan/zoom, compare, side view |
| Main window | `test_main_window.py` | shortcuts, tool rail, save/open project, missing photo, export, close prompt, drag & drop, render queue |
| Screen fitting | `test_layout.py` | window fits the available screen, panels never clip horizontally, tool rail reachable on small screens |
| Transparency | `test_transparency.py` | PNG/WebP alpha loading, hidden background never affects the subject, RGBA render, alpha export / JPEG fill, editor workflow, projects |
| Hygiene | `test_source_hygiene.py` | UTF-8 sources without double-encoded text, valid preset JSON |
| End to end | `test_integration.py` | real Depth Anything V2 + ViTMatte, full MVP workflow in the UI, depth race |

When you add a feature, add its test next to the matching row; when you fix a bug, add a test
that reproduces it first.
