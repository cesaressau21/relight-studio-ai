"""Project files (``.rsproj``).

A project is a ZIP container holding a versioned ``project.json`` plus binary
layers. The source photograph is referenced, never embedded or modified.

Layout::

    project.json        format, version, image reference, all parameters
    mask.png            relight mask (8-bit, edit resolution)            [optional]
    depth_offset.npy    depth corrections (float16, edit resolution)     [optional]
    disparity.npy       cached model output (float16) to skip inference  [optional]
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .document import Document
from .lights import Light, LightRig
from .settings import Adjustments, BackgroundSettings, DepthSettings, MaskSettings, RelightSettings

PROJECT_FORMAT = "relight-studio-project"
PROJECT_VERSION = 1
PROJECT_EXTENSION = ".rsproj"


class ProjectError(Exception):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _npy_load(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data), allow_pickle=False)


def save_project(doc: Document, path: str | Path) -> Path:
    path = Path(path)
    if doc.image_path is None:
        raise ProjectError("The document has no source image path")
    image_path = Path(doc.image_path).resolve()
    try:
        relative = os.path.relpath(image_path, path.parent.resolve())
    except ValueError:  # different drive on Windows
        relative = None
    manifest: dict[str, Any] = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "image": {
            "path": str(image_path),
            "relative_path": relative,
            "sha256": file_sha256(image_path) if image_path.exists() else None,
            "width": doc.width,
            "height": doc.height,
        },
        "lights": [l.to_dict() for l in doc.lights],
        "relight": doc.relight.to_dict(),
        "depth": doc.depth.to_dict(),
        "adjustments": doc.adjustments.to_dict(),
        "mask_settings": doc.mask_settings.to_dict(),
        "background": doc.background.to_dict(),
        "disparity_source": doc.disparity_source,
        "layers": {
            "mask": doc.mask is not None,
            "depth_offset": doc.depth_offset is not None,
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(manifest, indent=2))
        if doc.mask is not None:
            buf = io.BytesIO()
            Image.fromarray(doc.mask, "L").save(buf, format="PNG")
            zf.writestr("mask.png", buf.getvalue())
        if doc.depth_offset is not None:
            zf.writestr("depth_offset.npy", _npy_bytes(doc.depth_offset.astype(np.float16)))
        if doc.disparity is not None:
            zf.writestr("disparity.npy", _npy_bytes(doc.disparity.astype(np.float16)))
    os.replace(tmp, path)
    return path


@dataclass
class ProjectData:
    manifest: dict[str, Any]
    mask: np.ndarray | None
    depth_offset: np.ndarray | None
    disparity: np.ndarray | None

    def candidate_image_paths(self, project_path: Path) -> list[Path]:
        image = self.manifest["image"]
        paths = []
        if image.get("relative_path"):
            paths.append((project_path.parent / image["relative_path"]).resolve())
        paths.append(Path(image["path"]))
        return paths


def read_project(path: str | Path) -> ProjectData:
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read("project.json"))
            mask = None
            if "mask.png" in names:
                with Image.open(io.BytesIO(zf.read("mask.png"))) as im:
                    mask = np.array(im.convert("L"), dtype=np.uint8)
            offset = _npy_load(zf.read("depth_offset.npy")).astype(np.float32) if "depth_offset.npy" in names else None
            disp = _npy_load(zf.read("disparity.npy")).astype(np.float32) if "disparity.npy" in names else None
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Not a valid Relight Studio project: {exc}") from exc
    if manifest.get("format") != PROJECT_FORMAT:
        raise ProjectError("Unknown project format")
    if int(manifest.get("version", 0)) > PROJECT_VERSION:
        raise ProjectError(f"Project version {manifest['version']} is newer than this application supports")
    return ProjectData(manifest, mask, offset, disp)


def build_document(data: ProjectData, image: np.ndarray, image_path: Path,
                   opacity: np.ndarray | None = None) -> Document:
    m = data.manifest
    doc = Document(
        image=image,
        opacity=opacity,
        image_path=image_path,
        rig=LightRig([Light.from_dict(d) for d in m.get("lights", [])]),
        relight=RelightSettings.from_dict(m.get("relight")),
        depth=DepthSettings.from_dict(m.get("depth")),
        adjustments=Adjustments.from_dict(m.get("adjustments")),
        mask_settings=MaskSettings.from_dict(m.get("mask_settings")),
        background=BackgroundSettings.from_dict(m.get("background")),
    )
    w, h = doc.edit_size
    if data.mask is not None and data.mask.shape == (h, w):
        doc.mask = data.mask
    if data.depth_offset is not None and data.depth_offset.shape == (h, w):
        doc.depth_offset = data.depth_offset
    if data.disparity is not None:
        doc.disparity = data.disparity
        doc.disparity_source = m.get("disparity_source", "")
    return doc
