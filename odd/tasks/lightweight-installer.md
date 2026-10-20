# Lightweight Windows installer
Objective: a small setup.exe for GitHub Releases (under the 2 GB asset limit) that installs the app and downloads Python, PyTorch/CUDA and the other packages at install time.
Branch: feat/installer (from fix/hair-highlights b5592ab) · Delivery: ask-on-risk · Test runner: pytest (`.venv/Scripts/python -m pytest`)

## Specs
- S1 (verbatim, user's choice): "Instalador liviano (Recommended)" — "Un instalador pequeño en GitHub Releases que baja PyTorch/CUDA durante la instalación. Cabe en el límite de 2 GB, pero necesita internet al instalar."
- S2 Release context (verbatim choices): "MIT (Recommended)", "Beta pública 0.1.0 (Recommended)".
- S3 The installer picks the PyTorch build for the user's GPU: cu130 (NVIDIA compute capability ≥ 7.5 with a recent driver), cu126 (older NVIDIA GPUs), otherwise CPU.
- S4 Per-user install without administrator rights; Start-menu shortcut; uninstaller removes the downloaded environment; a "Repair" shortcut re-runs the environment setup.
- S5 Reproducible: pinned uv (SHA-256 verified), Python 3.12, torch 2.14.1 and a lock of every other runtime package; only PySide6-Essentials (the app uses QtCore/QtGui/QtWidgets).
- Assumption: Inno Setup 6 builds the setup.exe (installed with the user's approval on 2026-10-19).

## Tasks
| ID | Specs | Route | Status | Evidence |
|----|-------|-------|--------|----------|
| T1 | S3, S5 | inline | done | packaging/installer/install.ps1 (pinned uv + SHA-256, private Python 3.12 with --no-bin --no-registry, torch by GPU, lock); requirements-lock.txt (PySide6-Essentials only) |
| T2 | S1, S4 | inline | done | relight_studio.iss (per-user, RedirectionGuard=no, Start menu + Repair, long-path uninstall); build_installer.py; paths.is_installed marker |
| T3 | S1-S5 | inline | done | tests/test_installer.py (15); suite 434; real silent install on the dev PC: 2.2 MB setup, cu130, `torch 2.14.1+cu130 cuda True`, app opened sample 2 with depth + matting, repair no-op, uninstall left nothing. Commits 2099106, 5087856 |

Route rationale: one writer, sequential (the .iss runs T1's script).

## Log
- L1 (2026-10-19) User: "Sí, sigue con el paso 3: el instalador liviano."
- L2 Findings: Inno Setup absent (installed via winget on approval), uv 0.13.0 present; torch 2.14.1 Windows cp312 wheels exist for cu130, cu126 and cpu (no cu128); the uv cache already holds torch 2.14.1+cu130, so local tests can reuse it (`RELIGHT_UV_CACHE`).
- L3 Install tests found and fixed: (1) `uv python install` writes python.exe to `~/.local/bin` and registers `HKCU\Software\Python\Astral\CPython3.12.15`; a manual test overwrote the user's own entry (restored to `AppData\Roaming\uv\python\cpython-3.12.15-windows-x86_64-none`), so setup now passes `--no-bin --no-registry`. (2) os error 448: Inno 6.5+ RedirectionGuard is inherited by child processes and blocks uv's minor-version junction; `RedirectionGuard=no` (per-user installer, never elevated). (3) Windows PowerShell 5.1 strips double quotes from native arguments. (4) Inno cannot delete paths over 260 characters; uninstall runs `rmdir` with the `\\?\` prefix.
- L4 (2026-10-19) User: "Sí, sigue con el paso 4: prueba el instalador en modo CPU y sin caché." Result: `RELIGHT_TORCH_VARIANT=cpu`, no uv cache: install 86 s, env 1.0 GB + Python 68 MB, temporary cache removed, `torch 2.14.1+cpu cuda False`. First launch with an empty `RELIGHT_MODELS_DIR`: both models downloaded (194 MB), depth 0.4-0.6 s per pass, AI matting at 1024x700 in 1.5 s; uninstall left only the test's own models folder and setup log (outside the app folder).
- L5 Not verified: another PC, an older NVIDIA GPU, the full cu130 download without the uv cache (~3 GB). Polish noticed: CPU installs show the 'CUDA is not available' notice as a red error at every launch; httpx2/huggingface_hub log every HTTP request at INFO during model downloads.
