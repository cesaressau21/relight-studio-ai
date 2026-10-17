# Relight Studio AI — MVP
Objective: local Windows desktop editor that relights photos using monocular depth (Depth Anything V2) and a 2.5D lighting engine.
Branch: feat/relight-mvp · Delivery: ask-on-risk (local repo, no remote) · Test runner: pytest (`.venv/Scripts/python -m pytest`)

## Specs
- S1 MVP acceptance (verbatim): "Abrir una fotografía real." "Generar un mapa de profundidad utilizando un modelo auténtico." "Colocar al menos una luz con control X, Y y Z." "Modificar su intensidad, color y radio." "Ver una iluminación dependiente de la geometría estimada." "Mover la luz y observar una actualización interactiva." "Conservar la imagen original." "Exportar el resultado." "Ejecutarse localmente en Windows."
- S2 Depth analysis: automatic depth map, normals from depth, edge-preserving smoothing, depth visualization, manual depth correction, near/far distinction.
- S3 Lights: at least three independent lights with X, Y, Z, intensity, radius, RGB color, Kelvin, softness, diffusion, distance attenuation, enable toggle. Types: point, directional, ambient, soft area, rim. Diffuse + specular from estimated normals; screen-space shadows with stated limitations; partial reflectance/illumination separation.
- S4 Direct interaction: circular gizmo, mouse drag with live update, modifier + wheel adjusts depth or radius, side panel for precise values, per-light id color, optional lateral depth view, positions stable under zoom/pan/resize.
- S5 UI: dark professional layout — top bar (logo, open, save project, export, undo/redo, compare), left tools (select, lights, mask brush, depth brush, navigation/zoom), central canvas with before/after, right panel (lights list, properties, color, depth, ambient, layers/masks), bottom status (processing, resolution, GPU, render time, progress). No Adobe assets.
- S6 Editing: exposure, contrast, highlights, shadows, whites, blacks, saturation, white balance, gamma, ambient intensity, original/new light mix, editable masks, presets (Studio Portrait, Golden Hour, Cinematic Blue & Orange, Neon Cyberpunk, Soft Natural Light, Dramatic Side Light, Sunset, Moonlight) that set real engine parameters.
- S7 Performance: reduced-resolution preview, cached depth/normals, GPU rendering, no repeated inference, heavy work off the UI thread, full-resolution export render, CUDA auto-detect with CPU fallback + warning, single model load.
- S8 Non-destructive: original never modified; versioned project file with image reference, light params/positions, masks, depth edits, adjustments; undo/redo.
- S9 Export: PNG, JPEG, WebP; resolution, quality, color profile; no gizmos in output.
- S10 Rules: real functionality only, no invented deps/models, verified PyTorch/CUDA/GPU compatibility, license-compatible models, documented install/run commands, tested with real photos, clear separation of implemented vs approximated vs pending.

## Tasks
| ID | Specs | Route | Status | Evidence |
|----|-------|-------|--------|----------|
| T1 | S10 | inline | done | Python 3.12.15 venv (uv), torch 2.14.1+cu130; DAv2-Small Apache-2.0 verified on HF API |
| T2 | S2, S7 | inline | done | dc454c7 · ai/depth.py, rendering/scene.py; real inference 50-80 ms after warm-up |
| T3 | S3 | inline | done | dc454c7 · rendering/engine.py; tests/test_engine.py (geometry, shadow, rim, mask, chunking) |
| T4 | S4, S5 | inline | done | dc454c7 · ui/canvas.py, depth_profile.py, main_window.py; native-window screenshot verified |
| T5 | S6, S8, S9 | inline | done | dc454c7 · 8 presets, history, .rsproj, export PNG/JPEG/WebP + sRGB ICC (tests/test_io.py) |
| T6 | S1, S10 | inline | done | dc454c7 · tests/test_integration.py: real model + offscreen UI + full-res export + project |
| T7 | S7, S10 | inline | in progress | warm-up, off-screen gizmos, depth race fix, PyInstaller spec |

Route rationale: one model writes sequentially; no Writer-trigger reason (no parallel disjoint units launched together).

## Log
- L1 (2026-10-16) User request: build "RELIGHT STUDIO AI", a fully functional Windows desktop app inspired by Photoshop Relight (full brief in session; specs above quote its acceptance criteria). Reference images: dark editor with central canvas and side controls; cinematic teal/orange relit portrait.
- L2 Environment: only Python 3.15 installed; torch cu128 has no cp315 wheel, cu130 does but ecosystem wheels lag → chose uv-managed Python 3.12. RTX 5070 = Blackwell sm_120 → requires CUDA ≥ 12.8 wheels; using cu130 (driver UMD 13.4).
- L3 Decision: rendering engine implemented in PyTorch (CUDA, CPU fallback) instead of GLSL so preview and full-resolution export share one numerically testable code path; documented in docs/ARCHITECTURE.md.
- L4 Verification: 53 tests pass (unit + integration with DAv2-Small, offscreen UI, full-res export). Benchmarks RTX 5070: preview 1600x1093, 3 lights, 24 shadow steps = 54 ms; export 6000x4100 = 1.1 s, peak VRAM 2.76 GB. First CUDA use costs ~9 s (lazy kernel loading) -> background warm-up added.
- L5 Fixes found while testing: chunked export differed at chunk borders (3x3 shadow blur) -> 1-row halo; preset intensities overexposed -> recalibrated; normals produced silhouette halos -> self-guided smoothing + slope clamp; off-frame lights unreachable -> gizmos pinned to canvas edge; depth result could land on a newer document -> results bound to their document + rerun.
- L6 Gotcha: Path.read_text/write_text without encoding on Windows wrote cp1252/CRLF into sources; always pass encoding="utf-8" (fixed).
