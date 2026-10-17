"""Image loading/export and project files."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from relight_studio.core.document import Document
from relight_studio.core.image_io import ExportOptions, load_image, save_image
from relight_studio.core.lights import LightType
from relight_studio.core.project_io import (
    PROJECT_VERSION,
    ProjectError,
    build_document,
    read_project,
    save_project,
)


def test_load_image_applies_exif_orientation(tmp_path):
    arr = np.zeros((20, 40, 3), np.uint8)
    arr[:, :20] = 255
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90° clockwise on display
    path = tmp_path / "rot.jpg"
    im.save(path, exif=exif, quality=95)
    loaded = load_image(path)
    assert loaded.shape == (40, 20, 3)


def test_load_image_flattens_alpha(tmp_path):
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 128
    path = tmp_path / "a.png"
    Image.fromarray(rgba, "RGBA").save(path)
    loaded = load_image(path)
    assert loaded.shape == (8, 8, 3) and loaded.dtype == np.uint8


@pytest.mark.parametrize("fmt,ext", [("PNG", ".png"), ("JPEG", ".jpg"), ("WEBP", ".webp")])
def test_export_formats_size_and_profile(tmp_path, fmt, ext):
    rgb = (np.random.default_rng(1).random((60, 90, 3)) * 255).astype(np.uint8)
    path = save_image(rgb, tmp_path / f"out{ext}", ExportOptions(format=fmt, quality=90))
    with Image.open(path) as im:
        assert im.format == fmt
        assert im.size == (90, 60)
        assert im.info.get("icc_profile")


def test_export_without_profile(tmp_path):
    path = save_image(np.zeros((4, 4, 3), np.uint8), tmp_path / "x.png", ExportOptions(embed_srgb=False))
    with Image.open(path) as im:
        assert not im.info.get("icc_profile")


def test_export_target_size():
    assert ExportOptions(long_side=1000).target_size(4000, 3000) == (1000, 750)
    assert ExportOptions(long_side=None).target_size(4000, 3000) == (4000, 3000)
    assert ExportOptions(long_side=8000).target_size(4000, 3000) == (4000, 3000)


def _doc_with_layers(tmp_path) -> Document:
    image_path = tmp_path / "photo.png"
    Image.fromarray(np.full((120, 160, 3), 77, np.uint8)).save(image_path)
    doc = Document(image=load_image(image_path), image_path=image_path)
    doc.add_light(LightType.POINT, x=0.1, y=0.2, z=-0.4, intensity=2.5, color=(1.0, 0.5, 0.25))
    doc.add_light(LightType.RIM, kelvin=3000)
    doc.relight.original_light = 0.2
    doc.depth.relief = 1.3
    doc.adjustments.exposure = -0.5
    doc.mask_settings.invert = True
    doc.ensure_mask()[10:20, 10:20] = 0
    doc.ensure_depth_offset()[5, 5] = 0.25
    doc.disparity = np.random.default_rng(2).random((40, 52)).astype(np.float32)
    doc.disparity_source = "small/standard"
    return doc


def test_project_roundtrip(tmp_path):
    doc = _doc_with_layers(tmp_path)
    project = save_project(doc, tmp_path / "scene.rsproj")
    data = read_project(project)
    assert data.manifest["version"] == PROJECT_VERSION
    image_path = next(p for p in data.candidate_image_paths(project) if p.exists())
    again = build_document(data, load_image(image_path), image_path)
    assert [l.to_dict() for l in again.lights] == [l.to_dict() for l in doc.lights]
    assert again.relight == doc.relight and again.depth == doc.depth
    assert again.adjustments == doc.adjustments and again.mask_settings == doc.mask_settings
    assert np.array_equal(again.mask, doc.mask)
    assert again.depth_offset[5, 5] == pytest.approx(0.25, abs=1e-3)
    assert np.allclose(again.disparity, doc.disparity, atol=1e-3)
    assert again.disparity_source == "small/standard"


def test_project_references_but_never_modifies_the_photo(tmp_path):
    doc = _doc_with_layers(tmp_path)
    before = doc.image_path.read_bytes()
    project = save_project(doc, tmp_path / "scene.rsproj")
    assert doc.image_path.read_bytes() == before
    with zipfile.ZipFile(project) as zf:
        assert "photo.png" not in zf.namelist()
        manifest = json.loads(zf.read("project.json"))
    assert manifest["image"]["relative_path"] == "photo.png"
    assert len(manifest["image"]["sha256"]) == 64


def test_project_rejects_invalid_and_newer_files(tmp_path):
    bad = tmp_path / "bad.rsproj"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ProjectError):
        read_project(bad)
    newer = tmp_path / "newer.rsproj"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr("project.json", json.dumps({"format": "relight-studio-project", "version": PROJECT_VERSION + 1}))
    with pytest.raises(ProjectError):
        read_project(newer)
