# Feature status

Legend: **Implemented** = works and is covered by tests or verified in the app ·
**Approximation** = works, but by a deliberate simplification you should know about ·
**Pending** = not built yet.

## MVP acceptance criteria

| Criterion | Status | Evidence |
|---|---|---|
| Open a real photograph | Implemented | JPEG/PNG/WebP/TIFF/BMP, EXIF orientation, ICC → sRGB (`tests/test_io.py`) |
| Transparent images (PNG/WebP alpha) | Implemented | transparency kept and used as the exact subject outline; hidden colors under alpha never reach depth or lighting; shown over a checkerboard; PNG/WebP export keeps alpha, JPEG (or unchecked *Keep transparency*) fills with a chosen color (`tests/test_transparency.py`) |
| Depth map from a real model | Implemented | Depth Anything V2 Small via Transformers (`tests/test_integration.py`) |
| A light with X, Y, Z control | Implemented | canvas drag, `Ctrl`+wheel, side view, numeric fields |
| Intensity, color, radius | Implemented | panel + wheel modifiers (`tests/test_engine.py`) |
| Lighting depends on estimated geometry | Implemented | sphere vs. flat test, cast-shadow test |
| Interactive update while moving | Implemented | coalesced background rendering, ~30–90 ms per preview frame on RTX 5070 |
| Original image preserved | Implemented | read-only array, file never written (tests) |
| Export | Implemented | PNG/JPEG/WebP, resolution, quality, sRGB profile, full-resolution render |
| Runs locally on Windows | Implemented | Windows 11 + RTX 5070, no network after the first model download |

## Engine

| Feature | Status | Notes |
|---|---|---|
| Depth estimation | Implemented | Small (Apache-2.0) default; Base/Large (CC-BY-NC) optional; "High" detail = 756 px input + flip TTA |
| Normals from depth | Approximation | height-field gradients, orthographic camera, slope clamped at depth jumps |
| Edge-preserving smoothing | Implemented | joint guided upsampling to image edges + self-guided smoothing for normals |
| Subject silhouette edges | Implemented | subject found from the depth histogram; AI matte (ViTMatte, hair-level) or fast color matte; subject and background depth layers composited by the matte, so the background next to a subject is no longer lit as subject (halo 108 → 2 levels in `tests/test_edges.py`) |
| Multi-layer scenes | Approximation | only two layers (subject / background): parts of the subject that are as far as the threshold (e.g. a shoulder or scarf) can be split off; adjust **Subject depth** |
| Manual depth correction | Implemented | push/pull brush, relief, near/far remap, gamma, invert |
| Reflectance / illumination separation | Approximation | shading = depth-guided smoothed luminance; colored or hard original light is only partially removed |
| Point / area / directional / ambient / rim lights | Implemented | area light = 5 samples for shading and for shadows (soft lateral penumbra) |
| Kelvin, softness, diffusion, falloff, enable | Implemented | |
| Light targets (everything / subject only / background only) | Implemented | weighted by the subject matte, so hair and soft edges are shared smoothly; the subject still casts shadows from background-only lights; without a separated subject the target is ignored and the panel says so (`tests/test_light_targets.py`) |
| Diffuse + specular | Approximation | wrapped Lambert + Blinn-Phong with a single global material |
| Shadows | Approximation | screen-space ray marching through the depth map: objects are treated as infinitely thick height fields, nothing off-screen casts shadows, rays leaving the frame are unoccluded |
| Ambient occlusion | Approximation | depth-difference cavity term, not ray-traced |
| Unlimited lights | Implemented | the minimum required was three |

## Editor

| Feature | Status |
|---|---|
| Light gizmos with id colors, drag, nudge, depth bar, radius ring | Implemented |
| Lateral depth view (side profile, draggable) | Implemented |
| Zoom / pan / fit / 1:1, gizmos stable under zoom, pan, resize | Implemented |
| Before/after split and hold-to-compare | Implemented |
| Exposure, contrast, highlights, shadows, whites, blacks, saturation, white balance, gamma | Implemented |
| Original/new light mix, ambient intensity | Implemented |
| Relight mask (paint, erase, fill, clear, invert, overlay) | Implemented |
| Eight presets (Studio Portrait … Moonlight) setting engine parameters | Implemented |
| Undo/redo with drag merging | Implemented |
| Project save/open (`.rsproj`, versioned JSON + layers, cached depth) | Implemented |
| GPU status, VRAM, render time, progress in the status bar | Implemented |
| Fits small screens | Implemented: window sized to the screen (maximized below 1300×760 logical), resizable side panel, scrolling tool rail, compact top bar under 1200 px, nothing clipped down to 1280×640 (`tests/test_layout.py`) |
| CPU fallback with warning | Implemented (`RELIGHT_DEVICE=cpu` forces it) |
| Windows executable | Implemented: PyInstaller one-folder build (~3.2 GB with the CUDA runtime), verified launching and running depth on the GPU |

## Pending / ideas

* Semantic segmentation into more than two layers (several people/objects at different depths).
* Per-light painted masks (lights can already target subject or background; free-form masks are still global).
* Learned intrinsic decomposition or generative relighting for stronger removal of the original
  light — would replace the heuristic shading estimate.
* GLSL port of the renderer for >60 fps previews at 4K.
* Color-managed export to wide-gamut profiles (export is sRGB).
* 16-bit input and output (the pipeline currently loads 8-bit).
