"""Lightweight installer: pinned components, GPU-specific PyTorch builds and the installed layout."""

from __future__ import annotations

import glob
import importlib.metadata as md
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INSTALLER = ROOT / "packaging" / "installer"
SCRIPT = INSTALLER / "install.ps1"
POWERSHELL = shutil.which("powershell")


def lock() -> dict[str, str]:
    pins = {}
    for line in (INSTALLER / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[re.sub(r"[-_.]+", "-", name).lower()] = version
    return pins


def script_value(name: str) -> str:
    match = re.search(rf"^\${name} = '([^']+)'", SCRIPT.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, name
    return match.group(1)


# ------------------------------------------------------------------ pinned components
def test_lock_matches_the_tested_environment():
    # When the development environment changes, regenerate the lock (see its header).
    for name, version in lock().items():
        assert md.version(name) == version, f"{name}: lock {version}, environment {md.version(name)}"


def test_lock_ships_only_the_qt_modules_the_app_uses():
    pins = lock()
    assert "pyside6-essentials" in pins and "shiboken6" in pins
    assert not {"pyside6", "pyside6-addons", "pyside6-webengine", "torch"} & set(pins)
    used = set(re.findall(r"from PySide6\.(\w+)", "\n".join(p.read_text(encoding="utf-8")
                                                          for p in (ROOT / "relight_studio").rglob("*.py"))))
    assert used <= {"QtCore", "QtGui", "QtWidgets"}  # all in PySide6-Essentials


def test_script_pins_torch_python_and_uv():
    assert script_value("TorchVersion") == md.version("torch").split("+")[0]
    assert script_value("PythonVersion") == f"{sys.version_info.major}.{sys.version_info.minor}"
    assert re.fullmatch(r"[0-9a-f]{64}", script_value("UvSha256"))
    assert "Get-FileHash $Zip" in SCRIPT.read_text(encoding="utf-8")  # verified before use


def test_python_stays_private_to_the_installation():
    # uv would otherwise put python.exe in ~/.local/bin and register it in HKCU\Software\Python,
    # replacing the user's own entry for the same version.
    text = SCRIPT.read_text(encoding="utf-8")
    assert "'--no-bin', '--no-registry'" in text


def test_script_is_ascii_for_windows_powershell():
    SCRIPT.read_bytes().decode("ascii")  # PowerShell 5.1 reads BOM-less files in the ANSI code page


# ------------------------------------------------------------------ GPU detection
@pytest.mark.skipif(POWERSHELL is None, reason="needs Windows PowerShell")
@pytest.mark.parametrize(("gpu", "variant"), [
    ("12.0, 581.57", "cu130"),   # RTX 50 with a CUDA 13 driver
    ("8.6, 617.42", "cu130"),    # RTX 30
    ("8.6, 552.44", "cu126"),    # RTX 30 with an older driver
    ("6.1, 552.44", "cu126"),    # GTX 10: CUDA 13 dropped it
    ("12.0, 566.03", "cpu"),     # RTX 50 needs CUDA 12.8+: update the driver
    ("6.1, 470.10", "cpu"),      # driver too old for CUDA 12
])
def test_pytorch_build_follows_the_gpu(gpu, variant):
    out = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                          "-DetectOnly", "-GpuInfo", gpu], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == variant


@pytest.mark.skipif(POWERSHELL is None, reason="needs Windows PowerShell")
def test_variant_can_be_forced():
    out = subprocess.run([POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                          "-DetectOnly", "-Variant", "cpu", "-GpuInfo", "12.0, 581.57"],
                         capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "cpu"


# ------------------------------------------------------------------ installer script
def test_installer_sources_exist_and_shortcut_runs_the_app():
    iss = (INSTALLER / "relight_studio.iss").read_text(encoding="utf-8")
    sources = re.findall(r'^Source: "([^"]+)"', iss, re.MULTILINE)
    assert sources
    for source in sources:
        path = source.replace("{#Root}", str(ROOT)).replace("\\", "/")
        if not Path(path).is_absolute():
            path = str(INSTALLER / path)
        assert glob.glob(path), source
    assert 'Parameters: "-m relight_studio"' in iss and "PrivilegesRequired=lowest" in iss
    for folder in ("env", "python", "tools", "cache"):
        assert f'Name: "{{app}}\\{folder}"' in iss  # the uninstaller removes what setup downloaded


def test_installed_copy_keeps_models_in_user_data(tmp_path, monkeypatch):
    from relight_studio.core import paths

    monkeypatch.delenv("RELIGHT_MODELS_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setattr(paths, "resource_root", lambda: tmp_path / "app")
    (tmp_path / "app").mkdir()
    assert paths.models_dir() == tmp_path / "app" / "models"  # a checkout keeps them in the repo
    shutil.copyfile(INSTALLER / "installed.txt", tmp_path / "app" / paths.INSTALLED_MARKER)
    assert paths.is_installed()
    assert paths.models_dir() == tmp_path / "local" / "RelightStudioAI" / "models"
