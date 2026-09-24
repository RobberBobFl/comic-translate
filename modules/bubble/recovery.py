"""Experimental manual bubble recovery + expansion (no ML, no new deps).

Ports the validated recipe from ``scratch/bubble_research`` (REPORT.md):

recovery   interior mask via C (white-threshold connected components inside the
           detector bubble box), falling back to C2 (colour-adaptive) when the
           text is not contained, then expanding through the stroke to the
           outer contour (Cx / C2x) with an auto-estimated stroke width.
expansion  dilate the outer mask, repaint a flat fill and redraw the outline as
           a *morphological ring* (``expanded - erode(expanded, r)``). The ring
           hits the requested width exactly, unlike ``cv2.drawContours`` which
           quantises stroke width in ~2px steps (Stage 7 calibration).

Inputs are only what the editor already holds: page RGB, text bbox and the
optional detector bubble box. Nothing here touches the automatic pipeline.
"""
from __future__ import annotations

import cv2
import numpy as np

WHITE = 185          # "bubble interior" grey threshold
DARK_STROKE = 100    # strong-stroke grey threshold
STROKE_GRAY = 120    # outline detection limit (covers the anti-aliased edge)
COLOR_MAX_DIST = 95  # L1 distance to the estimated fill colour (C2)
ELL3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# ------------------------------------------------------------------ utilities
def inflate(xyxy, frac: float, min_px: int, shape=None) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    w, h = x2 - x1, y2 - y1
    dx = max(min_px, int(w * frac))
    dy = max(min_px, int(h * frac))
    x1 -= dx
    y1 -= dy
    x2 += dx
    y2 += dy
    if shape is not None:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(shape[1], x2)
        y2 = min(shape[0], y2)
    return x1, y1, x2, y2


def fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, contours, -1, 255, -1)
    return out


def _center(xyxy) -> tuple[int, int]:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    return (x1 + x2) // 2, (y1 + y2) // 2


def _pick_component(binary: np.ndarray, roi: tuple[int, int, int, int],
                    center_xy) -> np.ndarray | None:
    """Component containing the point (falls back to the largest)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return None
    x1, y1, _, _ = roi
    cx, cy = center_xy
    lx, ly = cx - x1, cy - y1
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = None
    for i in range(n - 1):
        bx = stats[i + 1, cv2.CC_STAT_LEFT]
        by = stats[i + 1, cv2.CC_STAT_TOP]
        bw = stats[i + 1, cv2.CC_STAT_WIDTH]
        bh = stats[i + 1, cv2.CC_STAT_HEIGHT]
        if bx <= lx < bx + bw and by <= ly < by + bh:
            if best is None or areas[i] > areas[best]:
                best = i
    if best is None:
        best = int(np.argmax(areas))
    return (lab == (best + 1)).astype(np.uint8)


def containment(mask: np.ndarray, text_xyxy) -> float:
    x1, y1, x2, y2 = (int(v) for v in text_xyxy)
    sub = (mask > 0)[y1:y2, x1:x2]
    return float(sub.mean()) if sub.size else 0.0


# ------------------------------------------------------------------ interior C
def _interior_roi(text_xyxy, bubble_xyxy, shape):
    if bubble_xyxy is None or np.array_equal(np.asarray(bubble_xyxy, dtype=int),
                                             np.asarray(text_xyxy, dtype=int)):
        # no detector bubble box (or a user-drawn box where bubble == text):
        # search a generous region around the text instead
        return inflate(text_xyxy, 0.6, 30, shape=shape)
    return inflate(bubble_xyxy, 0.12, 8, shape=shape)


def interior_white(gray: np.ndarray, text_xyxy, bubble_xyxy) -> np.ndarray | None:
    """C: threshold + connected components inside the ROI around the text."""
    roi = _interior_roi(text_xyxy, bubble_xyxy, gray.shape)
    x1, y1, x2, y2 = roi
    binw = (gray[y1:y2, x1:x2] >= WHITE).astype(np.uint8)
    comp = _pick_component(binw, roi, _center(text_xyxy))
    if comp is None:
        return None
    mask = np.zeros(gray.shape, np.uint8)
    mask[y1:y2, x1:x2] = comp
    mask = fill_holes(mask)
    return mask if mask.any() else None


# ------------------------------------------------------------------ interior C2
def _fill_color(rgb: np.ndarray, text_xyxy, bubble_xyxy, text_mask) -> np.ndarray:
    """Dominant quantised colour of the bubble interior (frame sampling)."""
    H, W = rgb.shape[:2]
    if bubble_xyxy is not None and not np.array_equal(
            np.asarray(bubble_xyxy, dtype=int), np.asarray(text_xyxy, dtype=int)):
        bx1, by1, bx2, by2 = (int(v) for v in bubble_xyxy)
        tx1, ty1, tx2, ty2 = inflate(text_xyxy, 0.10, 4, shape=(H, W))
        frame = np.zeros((H, W), np.uint8)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), 1, -1)
        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), 0, -1)
        px = rgb[frame > 0].reshape(-1, 3)
        if len(px) >= 100:
            q = (px.astype(np.int16) // 24) * 24
            keys, counts = np.unique(q, axis=0, return_counts=True)
            return keys[int(np.argmax(counts))].astype(np.int16)
    x1, y1, x2, y2 = inflate(text_xyxy, 0.10, 4, shape=(H, W))
    crop = rgb[y1:y2, x1:x2].reshape(-1, 3)
    if text_mask is not None and text_mask.any():
        tm = cv2.dilate((text_mask > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
        keep = tm[y1:y2, x1:x2].reshape(-1) == 0
    else:
        keep = np.ones(crop.shape[0], bool)
    px = crop[keep]
    if len(px) < 20:
        return np.array([255, 255, 255])
    q = (px.astype(np.int16) // 24) * 24
    keys, counts = np.unique(q, axis=0, return_counts=True)
    return keys[int(np.argmax(counts))].astype(np.int16)


def interior_color(rgb: np.ndarray, gray: np.ndarray, text_xyxy, bubble_xyxy,
                   text_mask) -> np.ndarray | None:
    """C2: colour-adaptive interior (white / cream / yellow bubbles)."""
    fill = _fill_color(rgb, text_xyxy, bubble_xyxy, text_mask)
    roi = _interior_roi(text_xyxy, bubble_xyxy, gray.shape)
    x1, y1, x2, y2 = roi
    dist = np.abs(rgb[y1:y2, x1:x2].astype(np.int16) - fill).sum(2)
    comp = _pick_component((dist <= COLOR_MAX_DIST).astype(np.uint8), roi,
                           _center(text_xyxy))
    if comp is None:
        return None
    mask = np.zeros(gray.shape, np.uint8)
    mask[y1:y2, x1:x2] = comp
    mask = fill_holes(mask)
    return mask if mask.any() else None


# ------------------------------------------------------------------- outer Cx
def measure_stroke_width(gray: np.ndarray, interior: np.ndarray,
                         max_r: int = 12) -> int:
    """Stroke width in px: consecutive dark rings just outside the interior.

    Successive 1px dilations walk outward from the interior boundary; the
    stroke is the run of rings whose mean grey stays below ``STROKE_GRAY``
    (the first ring is usually the anti-aliased stroke edge, hence a limit
    above pure black). Stage 7 measured 4-8px (median 5) this way.
    """
    cur = (interior > 0).astype(np.uint8)
    w = 0
    for _ in range(max_r):
        d = cv2.dilate(cur, ELL3)
        ring = (d > 0) & (cur == 0)
        if not ring.any():
            break
        if float(gray[ring].mean()) >= STROKE_GRAY:
            break
        w += 1
        cur = d
    return w


def expand_to_outer(interior: np.ndarray, k: int) -> np.ndarray:
    cur = (interior > 0).astype(np.uint8)
    for _ in range(max(0, k)):
        cur = cv2.dilate(cur, ELL3)
    return fill_holes(cur.astype(np.uint8) * 255)


def trim_fragments(outer: np.ndarray, text_xyxy, thresh: float = 0.4,
                   max_frag_frac: float = 0.4, min_cont: float = 0.9):
    """Drop mask regions beyond deep convexity-defect necks (tails, merged blobs).

    A deep defect marks a neck where the mask protrudes away from the body
    (a speech-bubble tail, or a neighbouring white region the recovery leaked
    into). The cut runs through the defect's deepest point, perpendicular to
    the hull edge; the side away from the mask centroid is discarded and left
    as original pixels (never expanded, never repainted).

    Guards: a candidate cut is skipped when it would remove more than
    ``max_frag_frac`` of the mask or drop the text containment below
    ``min_cont``. Returns ``(kept, fragments, stats)``; when nothing passes
    the guards the original mask comes back untouched.
    """
    mask = (outer > 0).astype(np.uint8)
    stats = {"trimmed": False, "fragments": 0.0, "containment": containment(mask, text_xyxy)}
    if mask.sum() == 0:
        return outer, np.zeros_like(mask), stats

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return outer, np.zeros_like(mask), stats
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c, returnPoints=False)
    if hull is None or len(hull) < 3:
        return outer, np.zeros_like(mask), stats
    defects = cv2.convexityDefects(c, hull)
    if defects is None:
        return outer, np.zeros_like(mask), stats

    H, W = mask.shape
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dmax = max(1.0, float(dt.max()))
    ys, xs = np.where(mask > 0)
    centre = (float(xs.mean()), float(ys.mean()))
    total = int(mask.sum())
    yy, xx = np.mgrid[0:H, 0:W]

    kept = mask.copy()
    fragments = np.zeros_like(mask)
    frag_px = 0
    for d in defects[:, 0]:
        depth = d[3] / 256.0
        if depth / dmax < thresh:
            continue
        p0 = c[int(d[0])].reshape(2).astype(float)
        p1 = c[int(d[1])].reshape(2).astype(float)
        far = c[int(d[2])].reshape(2).astype(float)
        u = p1 - p0
        un = float(np.linalg.norm(u))
        if un < 1e-6:
            continue
        u /= un

        def side(x, y):
            return u[0] * (x - far[0]) + u[1] * (y - far[1])

        sc = side(*centre)
        if sc == 0:
            continue
        frag = (mask > 0) & (np.sign(side(xx, yy)) != np.sign(sc))
        area = int(frag.sum())
        if area == 0 or area > max_frag_frac * total:
            continue
        trial = kept & (frag == 0)
        if containment(trial, text_xyxy) < min_cont:
            continue
        kept = trial
        fragments |= frag
        frag_px += area

    if frag_px == 0:
        return outer, np.zeros_like(mask), stats
    stats = {
        "trimmed": True,
        "fragments": round(100.0 * frag_px / total, 1),
        "containment": round(containment(kept, text_xyxy), 3),
    }
    return kept.astype(np.uint8) * 255, fragments, stats


def recover_bubble(rgb: np.ndarray, text_xyxy, bubble_xyxy=None,
                   text_mask=None) -> dict | None:
    """Recommended path: C -> Cx, falling back to C2 -> C2x.

    Returns ``{"interior", "outer", "fragments", "stroke_w", "method", "trim"}``
    or ``None`` when no plausible bubble was found. ``stroke_w`` is the measured
    outline width in px (0 when the bubble has no dark outline). ``outer`` is the
    body mask after tail/fragment trimming; ``fragments`` marks what was cut.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inner = interior_white(gray, text_xyxy, bubble_xyxy)
    if inner is not None and containment(inner, text_xyxy) < 0.5:
        inner = None  # the white blob found is not the bubble around this text

    if inner is not None:
        method = "C->Cx"
    else:
        inner = interior_color(rgb, gray, text_xyxy, bubble_xyxy, text_mask)
        method = "C2->C2x"
    if inner is None:
        return None

    stroke_w = measure_stroke_width(gray, inner)
    outer = expand_to_outer(inner, stroke_w)
    if not outer.any():
        return None
    outer, fragments, trim_stats = trim_fragments(outer, text_xyxy)
    return {
        "interior": inner,
        "outer": outer,
        "fragments": fragments,
        "stroke_w": stroke_w,
        "method": method,
        "trim": trim_stats,
    }


# ------------------------------------------------------------------ expansion
def expand_mask(outer: np.ndarray, px: int) -> np.ndarray:
    """Grow the bubble mask by ``px`` px (isotropic)."""
    if px <= 0:
        return outer.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    expanded = cv2.dilate(outer, kernel)
    expanded = cv2.morphologyEx(
        expanded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    return fill_holes(expanded)


def repaint(rgb: np.ndarray, expanded: np.ndarray, stroke_r: int,
            ss: int = 3, sigma: float = 0.8) -> np.ndarray:
    """Flat white fill + morphological-ring outline (variant B from Stage 7).

    The ring is ``expanded - erode(expanded, r)``: it reproduces the target
    width exactly, unlike ``cv2.drawContours`` (quantised in ~2px steps). The
    ring is computed at ``ss`` x resolution on the bubble crop and downsampled
    with INTER_AREA, then slightly blurred - this kills the binary staircase
    while keeping a crisp comic-style stroke.
    """
    out = rgb.copy()
    m = (expanded > 0)
    if not m.any():
        return out
    r = int(stroke_r)
    if r <= 0:
        out[m] = (255, 255, 255)
        return out  # no dark outline on this bubble: repaint the fill only

    x1, y1, x2, y2 = mask_bbox(expanded, pad=r + ss + 4, shape=rgb.shape[:2])
    sub = out[y1:y2, x1:x2]
    subm = m[y1:y2, x1:x2]
    sub[subm] = (255, 255, 255)

    h, w = subm.shape[:2]
    ms = cv2.resize(subm.astype(np.uint8), (w * ss, h * ss), interpolation=cv2.INTER_NEAREST)
    k = ss * r
    er = cv2.erode(ms, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    band = ((ms > 0) & (er == 0)).astype(np.float32)
    alpha = cv2.resize(band, (w, h), interpolation=cv2.INTER_AREA)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    a = alpha[..., None]
    sub[:] = (sub * (1 - a)).astype(np.uint8)
    return out


def mask_bbox(mask: np.ndarray, pad: int = 0, shape=None) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    if pad:
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad
    if shape is not None:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(shape[1], x2)
        y2 = min(shape[0], y2)
    return x1, y1, x2, y2
