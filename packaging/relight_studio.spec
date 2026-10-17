# PyInstaller spec for Relight Studio AI (one-folder build).
#   .venv/Scripts/python.exe -m PyInstaller packaging/relight_studio.spec --noconfirm
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent

hidden = (
    collect_submodules("transformers.models.depth_anything")
    + collect_submodules("transformers.models.dpt")
    + ["PIL.ImageCms"]
)

a = Analysis(
    [str(ROOT / "relight_studio" / "__main__.py")],
    pathex=[str(ROOT)],
    datas=[(str(ROOT / "presets"), "presets"), (str(ROOT / "assets"), "assets")],
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "IPython", "pytest", "torch.utils.tensorboard", "torchvision"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RelightStudioAI",
    icon=str(ROOT / "assets" / "icon.ico"),
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="RelightStudioAI", upx=False)
