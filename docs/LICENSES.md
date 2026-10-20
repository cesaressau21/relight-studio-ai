# Licenses

Relight Studio AI itself is released under the MIT License (`LICENSE`).

# Third-party components

| Component | Use | License |
|---|---|---|
| Depth Anything V2 **Small** (`depth-anything/Depth-Anything-V2-Small-hf`) | default depth model | Apache-2.0 |
| Depth Anything V2 **Base** / **Large** | optional, selectable in Scene → Model | **CC-BY-NC-4.0 — non-commercial use only** |
| ViTMatte Small (`hustvl/vitmatte-small-composition-1k`) | AI subject matte for silhouette edges | Apache-2.0 |
| PyTorch | tensors, CUDA | BSD-3-Clause |
| Hugging Face Transformers | model loading | Apache-2.0 |
| PySide6 / Qt 6 | user interface | LGPL-3.0 (dynamically linked) |
| NumPy | arrays | BSD-3-Clause |
| OpenCV (`opencv-python-headless`) | high-quality downscaling | Apache-2.0 |
| Pillow | image decoding/encoding, ICC profiles | MIT-CMU (HPND) |

Licenses were checked against the Hugging Face model cards (API `cardData.license`) on
2026-10-16. The application warns in the UI when a non-commercial checkpoint is selected.

## What the Windows build ships

The PyInstaller build (`packaging/relight_studio.spec`) writes next to `RelightStudioAI.exe`:

* `LICENSE.txt` — the application's MIT license, and `LICENSES.md` — this summary;
* `licenses/` — the license files of every bundled Python package (collected from the runtime
  dependency tree by `packaging/collect_licenses.py`), an index (`THIRD_PARTY_LICENSES.txt`) and
  the LGPL-3.0 and GPL-3.0 texts that Qt's LGPL terms require.

Qt / PySide6 are dynamically linked: their DLLs in `_internal/PySide6` and `_internal/shiboken6`
can be replaced by a compatible build. The CUDA runtime libraries inside PyTorch are NVIDIA
software redistributed with the PyTorch package under NVIDIA's CUDA license terms. Model weights
are not bundled; they are downloaded on first use.

All icons, the logo and the theme are original work drawn with QPainter; no Adobe or other
third-party graphic assets are included.
