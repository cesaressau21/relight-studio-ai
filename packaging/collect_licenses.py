"""Gather the license texts of everything the application ships with.

Used by the PyInstaller spec after the build, and testable on its own::

    .venv/Scripts/python.exe packaging/collect_licenses.py dist/RelightStudioAI/licenses

Walks the runtime dependency tree of the application's direct dependencies (build and
test tools are not part of it), copies each package's license, notice and copying files
to ``<dest>/<package>/`` and writes an index. Qt is LGPL-3.0, whose terms require the
LGPL and GPL texts next to the program: they are kept in ``packaging/licenses``.
"""

from __future__ import annotations

import importlib.metadata as md
import re
import shutil
import sys
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent
DIRECT = ["PySide6", "torch", "transformers", "safetensors", "numpy", "opencv-python-headless", "pillow"]
LICENSE_FILE = re.compile(r"(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", re.IGNORECASE)
COPYLEFT_TEXTS = ROOT / "packaging" / "licenses"
INDEX = "THIRD_PARTY_LICENSES.txt"


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_distributions(direct: list[str] = DIRECT) -> list[md.Distribution]:
    """Installed distributions needed at run time, without optional extras."""
    found: dict[str, md.Distribution] = {}
    pending = list(direct)
    while pending:
        name = _canonical(pending.pop())
        if name in found:
            continue
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue  # optional on this platform
        found[name] = dist
        for spec in dist.requires or []:
            req = Requirement(spec)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                pending.append(req.name)
    return sorted(found.values(), key=lambda d: _canonical(d.metadata["Name"]))


def license_of(dist: md.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression") or ""
    if expression:
        return expression
    text = (meta.get("License") or "").strip()
    if text and len(text) < 80 and "\n" not in text:
        return text
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    return ", ".join(classifiers) or "see files"


def collect(dest: Path) -> list[tuple[str, str, str]]:
    """Copy license files into ``dest`` and write the index. Returns (name, version, license)."""
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for dist in runtime_distributions():
        name = dist.metadata["Name"]
        target = dest / _canonical(name)
        for file in dist.files or []:
            path = str(file).replace("\\", "/")
            if ".dist-info/" in path and LICENSE_FILE.search(file.name):
                relative = path.split(".dist-info/", 1)[1]
                relative = relative.removeprefix("licenses/")
                out = target / relative
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dist.locate_file(file), out)
        rows.append((name, dist.version, license_of(dist)))
    for text in sorted(COPYLEFT_TEXTS.glob("*.txt")):
        shutil.copyfile(text, dest / text.name)
    lines = [
        "Relight Studio AI bundles the following third-party packages.",
        "Each folder holds the license files shipped with that package.",
        "",
        "Qt / PySide6 are used under the LGPL-3.0 (see LGPL-3.0.txt and GPL-3.0.txt). They are",
        "dynamically linked: their DLLs and Python modules in _internal/PySide6 and _internal/shiboken6",
        "can be replaced by a compatible build of the same version.",
        "",
        "The CUDA runtime libraries in _internal/torch/lib (cudart, cuBLAS, cuDNN, ...) are NVIDIA",
        "software redistributed as part of the PyTorch package under NVIDIA's CUDA license terms.",
        "",
        "AI model weights are not bundled; they are downloaded on first use under their own",
        "licenses (see LICENSES.md).",
        "",
    ]
    width = max(len(n) for n, _, _ in rows)
    lines += [f"{n:<{width}}  {v:<16}  {lic}" for n, v, lic in rows]
    (dest / INDEX).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    rows = collect(Path(sys.argv[1]))
    print(f"{len(rows)} packages")
