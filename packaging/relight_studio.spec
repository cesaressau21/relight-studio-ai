# PyInstaller spec for Relight Studio AI (one-folder build).
#   .venv/Scripts/python.exe -m PyInstaller packaging/relight_studio.spec --noconfirm
import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent

hidden = (
    collect_submodules("transformers.models.depth_anything")
    + collect_submodules("transformers.models.dpt")
    + collect_submodules("transformers.models.vitmatte")
    + ["PIL.ImageCms"]
)

a = Analysis(
    [str(ROOT / "relight_studio" / "__main__.py")],
    pathex=[str(ROOT)],
    # Only the icons: the logo kit in assets/brand is not needed at run time.
    datas=[(str(ROOT / "presets"), "presets"), (str(ROOT / "assets" / "icon.png"), "assets"),
           (str(ROOT / "assets" / "icon.ico"), "assets")],
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

# License texts next to the executable: the app's own, the third-party summary and every
# bundled package's license files (Qt's LGPL requires them to travel with the program).
APP = Path(DISTPATH) / "RelightStudioAI"
shutil.copyfile(ROOT / "LICENSE", APP / "LICENSE.txt")
shutil.copyfile(ROOT / "docs" / "LICENSES.md", APP / "LICENSES.md")
sys.path.insert(0, str(ROOT / "packaging"))
from collect_licenses import collect  # noqa: E402

shutil.rmtree(APP / "licenses", ignore_errors=True)
collect(APP / "licenses")
