"""Image I/O operations for the imkit module."""

from __future__ import annotations
import os
from io import BytesIO
import numpy as np
from PIL import Image
from .utils import ensure_uint8

# ComicTranslate stitches entire webtoons into single images that can far exceed
# PIL's default ~178M-pixel decompression-bomb limit. Disable the check so
# stitched/recovered projects load. This is a local desktop tool processing the
# user's own files.
Image.MAX_IMAGE_PIXELS = None


def read_image(path: str) -> np.ndarray:
    """Read an image file and return as RGB numpy array."""
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.array(im)
    return arr


def write_image(path: str, array: np.ndarray, quality: int | None = None,
                jpeg_options: dict | None = None) -> None:
    """Write a numpy array as an image file.

    ``jpeg_options`` carries source-faithful JPEG encoding parameters
    (qtables/subsampling) captured by read_jpeg_encode_options; when supplied
    for a .jpg/.jpeg target they replace Pillow's defaults.
    """
    im = Image.fromarray(ensure_uint8(array))
    save_kwargs: dict[str, object] = {}

    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        if jpeg_options:
            try:
                im.save(path, **jpeg_options)
                return
            except (ValueError, OSError):
                pass
        try:
            im.save(path, quality="keep", **save_kwargs)
            return
        except (ValueError, OSError):
            pass
    if ext == ".webp" and quality is not None:
        save_kwargs["quality"] = quality

    im.save(path, **save_kwargs)


def read_jpeg_encode_options(path: str) -> dict | None:
    """Capture a source JPEG's quantization tables and chroma subsampling so a
    re-encode can match the original quality instead of Pillow's default q75.

    Returns None for non-JPEG sources or when nothing usable was captured.
    """
    try:
        with Image.open(path) as im:
            if im.format != "JPEG":
                return None
            options: dict[str, object] = {}
            qtables = getattr(im, "quantization", None)
            if qtables:
                options["qtables"] = qtables
            try:
                from PIL import JpegImagePlugin

                sampling = JpegImagePlugin.get_sampling(im)
                if sampling in (0, 1, 2):
                    options["subsampling"] = sampling
            except Exception:
                pass
            return options or None
    except Exception:
        return None


def encode_image(array: np.ndarray, ext: str = ".png", **kwargs) -> bytes:
    """Encode a numpy array as image bytes."""
    if not ext.startswith('.'):
        ext = '.' + ext

    fmt = ext.lstrip('.').upper()
    im = Image.fromarray(ensure_uint8(array))
    buf = BytesIO()
    save_kwargs = {}

    fmt = "JPEG" if fmt == "JPG" else fmt
    if fmt == "JPEG":
        # Arrays have no original JPEG state, so Pillow's quality="keep"
        # always raises; an explicit quality is the only thing that applies.
        save_kwargs.setdefault("quality", kwargs.get("quality", 75))
        im.save(buf, format=fmt, **save_kwargs)
        return buf.getvalue()
    if fmt == "PNG":
        # Pillow uses 0 (no compression) to 9. Mirror cv2.IMWRITE_PNG_COMPRESSION default 3.
        save_kwargs.setdefault("compress_level", kwargs.get("compress_level", 3))
    im.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to numpy array."""
    im = Image.open(BytesIO(data))
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    arr = np.array(im)
    # Preserve single-channel grayscale arrays (ndim == 2).
    return arr
