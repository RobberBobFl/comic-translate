"""Export fidelity for regular (non-webtoon) pages.

Untouched pages must be exported bit-identical (source bytes copied), and
edited JPEG pages must be re-encoded with the source's quantization tables
and subsampling instead of Pillow's default q75.
"""
import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, JpegImagePlugin
from PySide6 import QtWidgets

import imkit as imk
from app.controllers.projects import ProjectController


@pytest.fixture(scope="module", autouse=True)
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_jpeg(path: str, quality=90, subsampling=0, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(96, 72, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=quality, subsampling=subsampling)
    with open(path, "rb") as f:
        return f.read()


def _make_worker(page_path: str, viewer_state=None, paint_overlay=None):
    main = SimpleNamespace()
    main.tr = lambda s: s
    main.webtoon_mode = False
    main.image_files = [page_path]
    main.image_patches = {}
    main.file_handler = SimpleNamespace(
        should_pre_materialize=lambda files: False,
        pre_materialize=lambda files: 0,
    )
    main.load_image = lambda p: np.array(Image.open(p).convert("RGB"))
    obj = SimpleNamespace(main=main)
    obj._build_export_page_name = ProjectController._build_export_page_name
    worker = ProjectController.save_and_make_worker.__get__(obj)
    state = {page_path: {"viewer_state": viewer_state or {}, "paint_overlay": paint_overlay}}
    return worker, state


def _run_export(worker, state, tmp_path):
    plan = [{"group_name": "G", "page_indices": [0], "output_path": str(tmp_path / "out.cbz")}]
    worker(plan, state)
    with zipfile.ZipFile(str(tmp_path / "out.cbz")) as z:
        names = z.namelist()
        assert len(names) == 1
        return names[0], z.read(names[0])


def test_untouched_page_is_copied_verbatim(tmp_path):
    src = str(tmp_path / "page.jpg")
    original = _make_jpeg(src)

    worker, state = _make_worker(src)
    name, data = _run_export(worker, state, tmp_path)

    assert name == "page.jpg"
    assert data == original


def test_untouched_png_is_copied_verbatim(tmp_path):
    src = str(tmp_path / "page.png")
    rng = np.random.default_rng(3)
    Image.fromarray(rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)).save(src)
    with open(src, "rb") as f:
        original = f.read()

    worker, state = _make_worker(src)
    name, data = _run_export(worker, state, tmp_path)

    assert name == "page.png"
    assert data == original


def test_edited_jpeg_keeps_source_encoding(tmp_path):
    src = str(tmp_path / "page.jpg")
    _make_jpeg(src, quality=90, subsampling=0)
    with Image.open(src) as im:
        qtables = im.quantization

    # A retouch touch anywhere forces the page through the render path.
    paint = np.zeros((96, 72, 4), dtype=np.uint8)
    paint[:10, :10, 3] = 255

    worker, state = _make_worker(src, paint_overlay=paint)
    _, data = _run_export(worker, state, tmp_path)

    with Image.open(io.BytesIO(data)) as im:
        assert im.format == "JPEG"
        assert im.quantization == qtables
        assert JpegImagePlugin.get_sampling(im) == 0


def test_read_jpeg_encode_options_roundtrip(tmp_path):
    src = str(tmp_path / "s.jpg")
    _make_jpeg(src, quality=90, subsampling=0)
    opts = imk.read_jpeg_encode_options(src)
    assert opts is not None
    assert opts["subsampling"] == 0
    assert "qtables" in opts

    arr = np.array(Image.open(src).convert("RGB"))
    out = str(tmp_path / "r.jpg")
    imk.write_image(out, arr, jpeg_options=opts)
    with Image.open(out) as im:
        assert im.quantization == opts["qtables"]
        assert JpegImagePlugin.get_sampling(im) == 0


def test_read_jpeg_encode_options_rejects_non_jpeg(tmp_path):
    png = str(tmp_path / "s.png")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(png)
    assert imk.read_jpeg_encode_options(png) is None
