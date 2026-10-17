# Architecture

```
relight_studio/
  app/        entry point, logging
  core/       Qt-free domain model: document, lights, settings, history, brushes,
              presets, image I/O, project files, paths
  ai/         depth estimation (Depth Anything V2 through Hugging Face Transformers)
  rendering/  device selection, GPU filters, 2.5D scene builder, relighting renderer
  ui/         PySide6 widgets: main window, canvas, panels, workers, theme, icons
presets/      lighting presets (JSON, engine parameters)
models/       downloaded model weights (git-ignored)
tests/        unit tests + integration tests (real model, offscreen UI)
```

Dependencies point inward: `ui → rendering/ai → core`. `core` has no Qt or PyTorch imports,
so the document model, history and file formats are testable on their own.

## Data flow

```
photo (uint8, read-only) ──► Depth Anything V2 ──► disparity (cached in the document/project)
           │                                              │
           └──────────────► build_scene ◄─────────────────┘   + depth settings + depth edits
                               │  image_lin, depth01, z, normals, shading S, cavity, mask
                               ▼
              Renderer.render(scene, lights, settings, adjustments) ──► sRGB uint8
```

* **Preview**: the photo is downscaled to 1600 px (CUDA) / 1024 px (CPU) once. A scene is
  built for it and cached in the render thread; it is rebuilt only when the geometry inputs
  change (depth settings, depth edits, new depth). Light or grading changes only re-run the
  renderer. While a light is being dragged and full frames take > 45 ms, frames are drawn at
  half resolution with half the shadow samples, then refined on release.
* **Export**: a fresh scene is built at the export resolution and rendered in row chunks (with a
  one-row halo so chunking is seamless) to bound GPU memory; on CUDA out-of-memory the export
  retries on the CPU.

## Silhouette edges

Depth Anything predicts at ~518 px and its edges are soft: upsampled, the subject's depth
bleeds several pixels into the background, which then glows like the subject and casts ghost
shadows. `rendering/edges.py` fixes this with two layers:

1. **Subject detection**: Otsu threshold on the depth histogram, accepted only when the histogram
   has a valley at the split (a real jump, not a sloped floor).
2. **Matte**: trimap from depth (unknown band = `edge_band`), then ViTMatte (`ai/matting.py`,
   computed in a background thread at preview resolution and guided-upsampled for export) or a fast
   color projection between local subject/background colors.
3. **Layers**: background pixels whose depth is pulled toward the subject are discarded; each
   layer is grown across the band from its reliable pixels (doubling-radius dilation + push-pull).
4. **Composite**: depth and normals are blended by the matte, so shading and shadows follow the
   photo's real outline, including semi-transparent hair.

Scene → **Subject edges** selects AI matting (default), Fast or Off, and exposes the subject
depth and band width. Display → **Subject matte** shows the result.

## Lighting model

All lighting happens in linear RGB.

```
S        original illumination: luminance smoothed with a depth-guided filter
A        albedo estimate = I / S^delighting
relit    = A * (original_light * S^delighting + ambient + Σ diffuse_i) + Σ specular_i
```

With no lights and `original_light = 1` the output equals the photo exactly (tested).

* **Normals** from the gradient of a self-guided (edge-preserving) smoothed depth, with the
  slope clamped at ~68° so depth jumps between objects do not create grazing normals.
* **Diffuse**: wrapped Lambert (softness) blended toward half-Lambert (diffusion), times
  `(1 + (d/radius)^2)^-falloff` with a smooth cut-off at 4 × radius.
* **Specular**: normalized Blinn-Phong, orthographic view vector.
* **Shadows**: per pixel, a ray toward the light is marched through the depth map in screen
  space (24 jittered steps by default, max reach 0.6 image heights). A sample is occluded when
  the depth map is in front of the ray; penumbra widens with distance and softness.
* **Light types**: point, soft area (5 samples on a disc + wider penumbra), directional
  (direction from the scene center toward the gizmo), ambient (modulated by screen-space
  cavity occlusion), rim (silhouette term from normals, masked to surfaces in front of the light).
* **Grading**: exposure and white balance in linear light, out-of-range highlights burn to
  white (identity below 1.0), then gamma, levels, shadows/highlights, contrast and
  saturation in display space.

## Why PyTorch instead of GLSL shaders

The brief suggested OpenGL/GLSL for the preview. The renderer is written with PyTorch tensor
operations instead because:

1. one implementation serves the interactive preview *and* the full-resolution export, so what
   you see is what you export;
2. it is numerically testable without an OpenGL context (see `tests/test_engine.py`);
3. CUDA is already required for the depth network, and the same code runs on the CPU fallback.

On an RTX 5070 a 1600 × 1093 preview with three lights and shadows renders in ~30–60 ms.
A GLSL port of `rendering/engine.py` remains possible if higher frame rates are needed.

## Threads

| Thread | Work |
|---|---|
| UI | Qt event loop, document edits, history |
| Render (`RenderService`) | scene building and rendering; requests are coalesced (latest wins) |
| Depth (`DepthWorker`) | model warm-up at launch, inference per photo / re-analysis |
| Export (`ExportWorker`) | full-resolution render and file encoding |

The depth model is a process-wide singleton guarded by a lock: each checkpoint loads once.

## Project file (`.rsproj`)

A ZIP with `project.json` (`format: relight-studio-project`, `version: 1`), plus optional
`mask.png`, `depth_offset.npy` and `disparity.npy`. The source photo is referenced by absolute
path, relative path and SHA-256, and is never modified or embedded. Newer versions are rejected
with a clear message; unknown fields are ignored for forward compatibility.
