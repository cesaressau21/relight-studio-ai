<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/png/horizontal/relight-horizontal-color-on-dark-1200.png">
    <img alt="Relight Studio AI" src="assets/brand/png/horizontal/relight-horizontal-color-on-light-1200.png" width="420">
  </picture>
</p>

# Relight Studio AI

Desktop photo relighting for Windows. Relight Studio AI estimates the scene geometry of a
single photograph with a monocular depth network (Depth Anything V2), builds a 2.5D
representation (depth, normals, approximate albedo and original shading) and lets you place
and drag lights directly on the image. Lighting reacts to the estimated shape: surfaces facing
a light brighten, surfaces facing away darken, and foreground objects cast screen-space
shadows onto what is behind them.

> Relighting a single photograph is an ill-posed problem. The geometry is *relative* depth
> estimated by a neural network, the projection is treated as orthographic, and nothing
> hidden or outside the frame is known. Results are plausible approximations, not physically
> accurate inverse rendering. See [docs/STATUS.md](docs/STATUS.md).

## Install with the setup (Windows)

Download `RelightStudioAI-<version>-beta-setup.exe` (about 2 MB) from the GitHub Releases page
and run it. No administrator rights are needed. Setup copies the application and then
downloads its components into the installation folder, with a window showing progress:
Python 3.12, PyTorch 2.14.1 for your GPU and the other packages (about 3 GB with an NVIDIA
GPU, less without one), so keep the computer online. The PyTorch build follows the GPU:

| GPU | PyTorch build |
|---|---|
| NVIDIA, compute capability 7.5 or newer (RTX 20 and later, incl. RTX 50) with driver 580+ | CUDA 13.0 (`cu130`) |
| Older NVIDIA (GTX 9/10, Titan V) or an older driver; not RTX 50 | CUDA 12.6 (`cu126`) |
| No NVIDIA GPU | CPU (slower) |

The Start menu gets **Relight Studio AI** and **Repair Relight Studio AI** (re-runs the download,
for example after a connection drop or a driver update). The depth and matting models (~130 MB)
download on first launch into `%LOCALAPPDATA%\RelightStudioAI\models`. Uninstall from Windows
settings removes the application and everything setup downloaded; models and settings stay.
Windows SmartScreen may warn that the publisher is unknown (the setup is not code-signed): choose
**More info → Run anyway**.

## Requirements (from source)

| | Tested |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 5070 (Blackwell, `sm_120`), driver with CUDA 13 runtime support |
| Python | 3.12 (3.11–3.13 supported; 3.14+ lacks some wheels) |
| PyTorch | 2.14.1 + CUDA 13.0 (`cu130`) |

RTX 50-series GPUs need a PyTorch build compiled for CUDA ≥ 12.8; older `cu118`/`cu121`/`cu124`
wheels do not contain `sm_120` kernels. Without a usable CUDA device the app falls back to the
CPU and says so in the status bar (expect seconds per frame instead of milliseconds).

## Install from source

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/Scripts/python.exe torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
```

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Plain `pip` works the same way (`.venv\Scripts\python -m pip install ...`). The depth model
(~100 MB) is downloaded from Hugging Face on first launch into `models/`.

## Run

```bash
.venv/Scripts/python.exe -m relight_studio
```

Optionally pass a photo or a `.rsproj` project as the first argument.

## Test

```bash
.venv/Scripts/python.exe -m pytest
```

Integration tests (real depth model, offscreen UI, full-resolution export) run when the model
is cached and a photo exists in `samples/` (or `RELIGHT_TEST_IMAGE` points to one).
Run only the fast unit tests with `-m "not integration"`. See [docs/TESTING.md](docs/TESTING.md)
for what each test file protects.

## Build a Windows executable

```bash
.venv/Scripts/python.exe -m pip install pyinstaller
```

```bash
.venv/Scripts/python.exe -m PyInstaller packaging/relight_studio.spec --noconfirm
```

The output is `dist/RelightStudioAI/RelightStudioAI.exe` (a folder build: the CUDA runtime
alone is several GB). Model weights are not bundled; they download on first run.

## Build the setup (lightweight installer)

Needs [Inno Setup 6](https://jrsoftware.org/isinfo.php):

```bash
winget install JRSoftware.InnoSetup
```

```bash
.venv/Scripts/python.exe packaging/build_installer.py
```

The output is `dist/installer/RelightStudioAI-<version>-beta-setup.exe`. It holds the source,
presets, assets and licenses; `packaging/installer/install.ps1` downloads a pinned uv
(SHA-256 checked), Python, the PyTorch build for the GPU and the exact packages in
`packaging/installer/requirements-lock.txt`. When the development environment's packages change,
update the lock (`tests/test_installer.py` fails until they match).

## Using the editor

| Action | How |
|---|---|
| Open photo / project | `Ctrl+O`, or drop a file on the window (PNG/WebP transparency is kept) |
| Place a light | Pick a light tool on the left (`P` point, `A` area, `L` directional, `R` rim) and click the photo |
| Move a light | Drag its gizmo (`V` select tool); arrows nudge, `Shift` for bigger steps |
| Light depth (Z) | `Ctrl` + wheel over the canvas, or drag the dot in **Lights → Side view** |
| Radius / intensity | `Shift` + wheel / `Alt` + wheel |
| Light patterns (gobos) | **Lights → Pattern**: blinds, window panes, leaves, dots or a custom image (white lets light through); size, rotation, focus blur, strength |
| Replace the background | **Backdrop → Choose image…**, then **Match lighting to background** for a key/fill/rim rig that follows the new background's light; blur and brightness for the backdrop |
| Remove the original light's color | **Scene → Light color** (0–1) takes a teal/orange cast out of the subject so new lights and backgrounds are not tinted by it; **Pick neutral** and click something white or gray to also remove an overall cast (**Match lighting to background** turns it on) |
| Light only the subject or the background | **Lights → Affects**: Everything, Subject only (S badge) or Background only (B badge) |
| Compare | `C` toggles the before/after split (drag the divider); hold `\` to see the original |
| Mask brush | `B`; drag paints where relighting applies, `Alt`/right-drag erases, `[` `]` size |
| Depth brush | `D`; drag pushes surfaces away, `Alt`/right-drag pulls them closer |
| Inspect geometry | **Scene → Display**: depth map, normals, albedo, original shading, lighting only, occlusion |
| Undo / redo | `Ctrl+Z` / `Ctrl+Y` |
| Save project | `Ctrl+S` (`.rsproj`) |
| Export | `Ctrl+E`: PNG, JPEG or WebP, chosen resolution, embedded sRGB profile |

Documentation: [architecture](docs/ARCHITECTURE.md) · [feature status and limitations](docs/STATUS.md) ·
[third-party licenses](docs/LICENSES.md).

## License

Relight Studio AI is released under the [MIT License](LICENSE). Third-party components keep their
own licenses ([summary](docs/LICENSES.md)); the Windows build ships every bundled package's license
files in `licenses/`. The optional Depth Anything V2 Base/Large models are non-commercial
(CC-BY-NC-4.0) and are never downloaded unless selected.
