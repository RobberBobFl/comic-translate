"""Experimental manual bubble recovery + expansion (no ML, no new deps).

Ports the validated recipe from ``scratch/bubble_research`` (REPORT.md):

recovery   interior mask via C (white-threshold connected components inside the
           detector bubble box), falling back to C2 (colour-adaptive) when the
           text is not contained, then expanding through the stroke to the
           outer contour (Cx / C2x) with an auto-estimated stroke width.
expansion  dilate the outer mask, flat-fill the interior and transplant the
           original outline outward: the new edge ring is a copy of the true
           edge strip, shifted along the boundary normals by the expansion
           distance, so width / colour / anti-aliasing match the original
           balloon instead of a uniform synthetic band.

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

# Safety limits for the recovered region (see plausible_size / measure_stroke_width).
# A balloon interior never dwarfs its own text/box, and a balloon always has an
# ink outline; without these guards a white page margin is mistaken for a bubble
# and the expander floods the whole panel with white.
MAX_INNER_FRAC = 8.0      # interior area / text-box area
MAX_INNER_VS_BOX = 1.8    # interior area / detector bubble-box area
FALLBACK_STROKE = 4       # width when the outline merges into surrounding ink
MIN_STROKE = 2            # outline thinner than this -> not a balloon we can repaint
MAX_STROKE = 12           # G-pen outlines are much wider than the old 6px cap
MIN_OUTLINE_COVER = 0.55  # share of the boundary that must sit on ink
LIGHT_EDGE = 200          # grey here counts as background beyond an outline
REL_STROKE_TOL = 55       # background this much lighter than the stroke ends it
MIN_REL_RUN = 3           # need some ink on record before trusting that rule
MAX_HALO = 3              # a.a. pixels allowed between the ink and the light edge
NEAR_WHITE = 235          # interior at least this light stays pure white on repaint
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


def smooth_mask(mask: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    """Smooth the contour of a binary mask using a continuous Signed Distance Field (SDF).

    Unlike polygon vertex filtering, which must rasterize smoothed coordinates
    back to a discrete integer grid (re-introducing staircase steps and pinch
    artifacts), SDF computes the signed distance field (positive inside, negative
    outside), filters the continuous surface with a 2D Gaussian, and takes the
    zero level set.  This yields perfectly organic, smooth boundaries without
    corners or pixel-grid aliasing.
    """
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return mask.copy()
    dt_in = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    dt_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    sdf = dt_in - dt_out
    sdf_smooth = cv2.GaussianBlur(sdf, (0, 0), float(sigma))
    out = (sdf_smooth >= 0.0).astype(np.uint8) * 255
    return fill_holes(out)


def _center(xyxy) -> tuple[int, int]:
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    return (x1 + x2) // 2, (y1 + y2) // 2


def _pick_component(binary: np.ndarray, roi: tuple[int, int, int, int],
                    center_xy, text_xyxy=None) -> np.ndarray | None:
    """Component that best *covers the text* inside the ROI.

    A balloon interior surrounds its text, so the right component is the one
    covering most of the light pixels inside the text box. Picking instead the
    largest component whose bounding box merely *contains* the text centre is
    what used to select the light page margin wrapping the balloon (its bbox
    spans the whole ROI while its pixels are all outside the text) and fed the
    expander a page-wide mask. The centre test is kept as a cheap gate; the
    largest-area fallback only fires when no bbox contains the centre.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return None
    x1, y1, _, _ = roi
    cx, cy = center_xy
    lx, ly = cx - x1, cy - y1
    areas = stats[1:, cv2.CC_STAT_AREA]

    # reference = every candidate (light) pixel inside the text box, ROI-local
    ref = 0.0
    rx1 = ry1 = rx2 = ry2 = 0
    if text_xyxy is not None:
        tx1, ty1, tx2, ty2 = (int(v) for v in text_xyxy)
        rx1, ry1 = max(0, tx1 - x1), max(0, ty1 - y1)
        rx2 = min(binary.shape[1], tx2 - x1)
        ry2 = min(binary.shape[0], ty2 - y1)
        if rx2 > rx1 and ry2 > ry1:
            ref = float(binary[ry1:ry2, rx1:rx2].sum())

    best = None
    best_score = None
    for i in range(n - 1):
        bx = int(stats[i + 1, cv2.CC_STAT_LEFT])
        by = int(stats[i + 1, cv2.CC_STAT_TOP])
        bw = int(stats[i + 1, cv2.CC_STAT_WIDTH])
        bh = int(stats[i + 1, cv2.CC_STAT_HEIGHT])
        if not (bx <= lx < bx + bw and by <= ly < by + bh):
            continue
        if ref > 0:
            inside = int((lab[ry1:ry2, rx1:rx2] == (i + 1)).sum())
            score = (inside / ref, int(areas[i]))
        else:
            score = (-1.0, int(areas[i]))
        if best_score is None or score > best_score:
            best_score, best = score, i
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


def _union_roi(text_xyxy, others, shape, pad: int = 30, bubble_xyxy=None):
    """ROI covering the selected text plus nearby texts (for merged splits).

    Starts from the standard bubble-box ROI so nothing is lost when no
    neighbour is close, then grows to include nearby texts.
    """
    if bubble_xyxy is not None:
        bx1, by1, bx2, by2 = _interior_roi(text_xyxy, bubble_xyxy, shape)
        xs = [bx1, bx2]
        ys = [by1, by2]
    else:
        xs = [text_xyxy[0], text_xyxy[2]]
        ys = [text_xyxy[1], text_xyxy[3]]
    w = text_xyxy[2] - text_xyxy[0]
    h = text_xyxy[3] - text_xyxy[1]
    cx = (text_xyxy[0] + text_xyxy[2]) / 2
    cy = (text_xyxy[1] + text_xyxy[3]) / 2
    for o in others:
        ox = (o[0] + o[2]) / 2
        oy = (o[1] + o[3]) / 2
        if abs(ox - cx) < 2 * w + 120 and abs(oy - cy) < 2 * h + 120:
            xs += [o[0], o[2]]
            ys += [o[1], o[3]]
    x1, y1 = max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad)
    x2, y2 = min(shape[1], int(max(xs)) + pad), min(shape[0], int(max(ys)) + pad)
    return (x1, y1, x2, y2)


def interior_white(gray: np.ndarray, text_xyxy, bubble_xyxy,
                   roi=None) -> np.ndarray | None:
    """C: threshold + connected components inside the ROI around the text."""
    if roi is None:
        roi = _interior_roi(text_xyxy, bubble_xyxy, gray.shape)
    x1, y1, x2, y2 = roi
    binw = (gray[y1:y2, x1:x2] >= WHITE).astype(np.uint8)
    comp = _pick_component(binw, roi, _center(text_xyxy), text_xyxy)
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
                   text_mask, roi=None) -> np.ndarray | None:
    """C2: colour-adaptive interior (white / cream / yellow bubbles)."""
    fill = _fill_color(rgb, text_xyxy, bubble_xyxy, text_mask)
    if roi is None:
        roi = _interior_roi(text_xyxy, bubble_xyxy, gray.shape)
    x1, y1, x2, y2 = roi
    dist = np.abs(rgb[y1:y2, x1:x2].astype(np.int16) - fill).sum(2)
    comp = _pick_component((dist <= COLOR_MAX_DIST).astype(np.uint8), roi,
                           _center(text_xyxy), text_xyxy)
    if comp is None:
        return None
    mask = np.zeros(gray.shape, np.uint8)
    mask[y1:y2, x1:x2] = comp
    mask = fill_holes(mask)
    return mask if mask.any() else None


# ------------------------------------------------------------------- outer Cx
def _stroke_profile(gray: np.ndarray, interior: np.ndarray,
                    max_r: int = MAX_STROKE,
                    ignore_dt: np.ndarray | None = None,
                    ignore_r: float = 4.0) -> dict:
    """Sample the outline along the interior boundary normals.

    From evenly arc-length-resampled boundary points a short walk goes outward
    along the local normal, skipping the anti-aliased shoulder
    (grey ``[STROKE_GRAY, 200)`` within 2 steps) and counting the run of ink.
    Half a pixel is added when the run stops on a soft outer edge, so a
    4px+a.a. stroke measures ~4.5 instead of snapping.

    A walk only counts towards the width when it ends in *background*: either a
    light pixel (``LIGHT_EDGE``) or a pixel clearly lighter than the stroke
    itself (``ink_ref + REL_STROKE_TOL``). That second rule is what tells a
    black outline apart from the dark artwork standing behind it - without it
    the walk slides off the outline into brown/blue backdrop and reports a
    10px "stroke" for a 4px line. Runs that end in a second dark band (two
    outlines touching) are dropped for the same reason. Runs that *do* contact
    ink still count towards ``coverage`` (the share of the boundary that sits on
    ink), so a balloon over a black panel is not punished.

    ``ignore_dt`` is an optional distance transform of "cut" pixels (e.g. the
    watershed seam where a merged bubble was split from its neighbour):
    sample points closer than ``ignore_r`` to a cut are skipped entirely, so
    the seam - which has no ink by construction - does not drag ``coverage``
    down and the split balloon is judged by its true outline only.
    """
    prof = {"n": 0, "runs": np.empty(0), "clean": np.empty(0), "blocked": 0,
            "coverage": 0.0, "med": 0.0, "p25": 0.0}
    m = (interior > 0).astype(np.uint8)
    if not m.any():
        return prof
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return prof
    c = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float32)
    if len(c) < 12:
        return prof

    ys, xs = np.where(m > 0)
    cen = np.array([float(xs.mean()), float(ys.mean())], np.float32)

    loop = np.vstack([c, c[:1]])
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 16:
        return prof
    n = int(np.clip(total, 48, 480))
    si = np.linspace(0.0, total, n, endpoint=False)
    px = np.interp(si, s, loop[:, 0])
    py = np.interp(si, s, loop[:, 1])

    # smooth the sampled contour so the tangent (and thus the normal) is stable
    k = 2
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    px = np.convolve(np.concatenate([px[-k:], px, px[:k]]), ker, "same")[k:-k]
    py = np.convolve(np.concatenate([py[-k:], py, py[:k]]), ker, "same")[k:-k]

    H, W = gray.shape
    if ignore_dt is not None:
        ign = ignore_dt
    else:
        ign = None
    hits = []
    clean = []
    blocked = 0
    ink_contact = 0
    counted = 0
    for i in range(n):
        if ign is not None:
            xi = int(round(px[i]))
            yi = int(round(py[i]))
            if 0 <= xi < ign.shape[1] and 0 <= yi < ign.shape[0]:
                if float(ign[yi, xi]) < ignore_r:
                    continue
        counted += 1
        j, h = (i + 1) % n, (i - 1) % n
        tx, ty = float(px[j] - px[h]), float(py[j] - py[h])
        nt = float(np.hypot(tx, ty))
        if nt < 1e-3:
            continue
        nx, ny = -ty / nt, tx / nt
        if nx * (px[i] - cen[0]) + ny * (py[i] - cen[1]) < 0:
            nx, ny = -nx, -ny
        run = 0.0
        halo = 0
        saw_ink = False
        ink_ref = 255          # darkest of the first ink pixels (the stroke core)
        n_ink = 0
        hit_clean = False       # ended on a genuinely light pixel
        verdict = "blocked"      # never reached background
        for step in range(1, max_r + 1):
            x = int(round(px[i] + nx * step))
            y = int(round(py[i] + ny * step))
            if not (0 <= x < W and 0 <= y < H):
                verdict = "ok"   # left the image: it was background
                break
            g = gray[y, x]
            if n_ink >= MIN_REL_RUN and g > ink_ref + REL_STROKE_TOL:
                # back to background that is clearly lighter than the stroke:
                # a dark *art* backdrop behind the outline is not stroke width
                verdict = "ok"
                if g < LIGHT_EDGE:
                    run += 0.5   # that pixel was the soft outer edge, not backdrop
                break
            if g >= LIGHT_EDGE:
                verdict = "ok"
                hit_clean = True
                break
            if g < STROKE_GRAY:
                if halo:
                    # a second dark band right behind the first one: two
                    # outlines touching - not a measurable stroke here
                    verdict = "drop"
                    break
                saw_ink = True
                run += 1.0
                n_ink += 1
                if n_ink <= 3:
                    ink_ref = min(ink_ref, g)
            elif run == 0.0 and step <= 2:
                continue          # anti-aliased inner shoulder: keep walking
            else:
                halo += 1         # soft outer edge of the stroke
                if halo > MAX_HALO:
                    verdict = "drop"  # never got to a light pixel
                    break
        if saw_ink:
            ink_contact += 1
        if verdict == "blocked":
            blocked += 1
            continue
        if verdict == "drop" or run <= 0.0:
            continue
        hits.append(run + (0.5 if halo else 0.0))
        clean.append(hit_clean)

    hits = np.asarray(hits)
    prof["runs"] = hits
    prof["clean"] = np.asarray(clean, dtype=bool)
    prof["blocked"] = blocked
    prof["n"] = counted
    prof["coverage"] = float(ink_contact) / counted if counted else 0.0
    if hits.size:
        prof["med"] = float(np.median(hits))
        prof["p25"] = float(np.percentile(hits, 25))
    return prof


def measure_stroke_width(gray: np.ndarray, interior: np.ndarray,
                         max_r: int = MAX_STROKE,
                         ignore_dt: np.ndarray | None = None,
                         ignore_r: float = 4.0) -> int:
    """Outline width in px (median of the normal runs).

    Samples whose walk ends in a *light* background are the unambiguous ones,
    so they win when there are enough of them; samples that had to stop on a
    relative-darkness rule (dark artwork behind the outline) are only a
    fallback, because their length includes a bit of that backdrop. The width
    is the median - robust against tail tips and jagged corners, unlike the old
    ring count which accumulated one full ring per step. Returns 0 when the
    outline is absent on too much of the boundary (``MIN_OUTLINE_COVER``), and
    ``FALLBACK_STROKE`` when there are too few usable samples to measure.
    """
    prof = _stroke_profile(gray, interior, max_r, ignore_dt, ignore_r)
    runs = prof["runs"]
    n = prof["n"]
    if prof["coverage"] < MIN_OUTLINE_COVER:
        return 0
    need = max(6, int(0.15 * n))
    src = runs
    if prof["clean"].size and int(prof["clean"].sum()) >= need:
        src = runs[prof["clean"]]
    if src.size < need:
        return FALLBACK_STROKE
    return int(np.clip(round(float(np.median(src))), 0, max_r))


def expand_to_outer(interior: np.ndarray, k: int, gray: np.ndarray | None = None) -> np.ndarray:
    """Expand interior mask to outer boundary of the outline stroke.

    If ``gray`` is provided, uses morphological gradient watershed to snap
    the boundary precisely to the outer edge of the ink stroke, followed by
    SDF smoothing. Otherwise falls back to isotropic dilation.
    """
    m = (interior > 0).astype(np.uint8)
    r = int(max(1, round(k)))
    if gray is not None and m.any():
        pad = min(3, max(1, r))
        band_in = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*max(0, r - pad)+1, 2*max(0, r - pad)+1)))
        band_out = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(r + pad)+1, 2*(r + pad)+1)))
        ys, xs = np.where(band_out > 0)
        if len(xs) > 0:
            x1, y1 = max(0, xs.min() - 5), max(0, ys.min() - 5)
            x2, y2 = min(gray.shape[1], xs.max() + 6), min(gray.shape[0], ys.max() + 6)
            sub_g = gray[y1:y2, x1:x2]
            sub_in = band_in[y1:y2, x1:x2]
            sub_out = band_out[y1:y2, x1:x2]
            grad = cv2.morphologyEx(sub_g, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            markers = np.zeros_like(sub_g, dtype=np.int32)
            markers[sub_in > 0] = 1
            markers[sub_out == 0] = 2
            cv2.watershed(cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR), markers)
            snapped = np.zeros_like(gray, dtype=np.uint8)
            snapped[y1:y2, x1:x2] = (markers == 1).astype(np.uint8) * 255
            return smooth_mask(fill_holes(snapped))

    cur = m
    for _ in range(max(0, k)):
        cur = cv2.dilate(cur, ELL3)
    return smooth_mask(fill_holes(cur.astype(np.uint8) * 255))


def _max_cross_width(frag: np.ndarray, origin: np.ndarray, axis: np.ndarray,
                     bins: int = 16) -> float:
    """Max cross-section width of `frag` along `axis`, in pixels. Bin counts
    are bin-length normalised: raw counts measure bin AREA, which skews the
    frag-vs-mask comparison when their axial depths differ."""
    ys, xs = np.where(frag)
    if ys.size == 0:
        return 0.0
    t = (xs - origin[0]) * axis[0] + (ys - origin[1]) * axis[1]
    t = t - t.min()
    depth = max(1.0, float(t.max()))
    idx = np.clip((t / depth * bins).astype(int), 0, bins - 1)
    return float(np.bincount(idx, minlength=bins).max()) / (depth / bins)


def _narrows_to_tip(frag: np.ndarray, origin: np.ndarray, axis: np.ndarray,
                    bins: int = 12) -> bool:
    """True when `frag` is tail-like: its widest cross-section sits AT the
    cut line and the width tapers toward the tip.

    A real speech-bubble tail is widest at the cut and narrows to the tip.
    A body wedge cut by an angular notch is widest in the MIDDLE of its
    profile (the cut runs along the balloon body, page 19, bubble
    "Я же говорю...") - such pieces stay in the mask.

    `axis` points from the cut line toward the piece tip.
    """
    ys, xs = np.where(frag)
    if ys.size == 0:
        return True
    dx = xs - origin[0]
    dy = ys - origin[1]
    t = dx * axis[0] + dy * axis[1]
    t = t - t.min()
    depth = float(t.max())
    if depth < 1.0:
        return True  # degenerate sliver: cut as before
    idx = np.clip((t / depth * bins).astype(int), 0, bins - 1)
    widths = np.bincount(idx, minlength=bins).astype(float)
    if widths.argmax() >= bins // 3:
        return False  # widest away from the cut: a wedge along the body
    q = max(1, bins // 4)
    base = widths[:q].mean()
    tip = widths[-q:].mean()
    return tip < base * 0.65


def trim_fragments(outer: np.ndarray, text_xyxy, thresh: float = 0.22,
                   max_frag_frac: float = 0.4, min_cont: float = 0.85):
    """Drop mask regions beyond deep convexity-defect necks (tails, merged blobs).

    A deep defect marks a neck where the mask protrudes away from the body
    (a speech-bubble tail, or a neighbouring white region the recovery leaked
    into). Tail trimming cuts across the neck between the defect notches
    (paired defects at the tail tip, or perpendicular to the protrusion direction
    for deep unpaired defects). The side away from the mask centroid is discarded
    and left as original pixels (never expanded, never repainted).

    Guards: a candidate cut is skipped when it would remove more than
    ``max_frag_frac`` of the mask or drop the text containment below
    ``min_cont``. Returns ``(kept, fragments, stats)``; when nothing passes
    the guards the original mask comes back untouched.
    """
    mask = (outer > 0).astype(np.uint8)
    orig_cont = containment(mask, text_xyxy)
    stats = {"trimmed": False, "fragments": 0.0, "containment": orig_cont}
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
    centre = np.array([float(xs.mean()), float(ys.mean())])
    total = int(mask.sum())
    yy, xx = np.mgrid[0:H, 0:W]

    kept = mask.copy()
    fragments = np.zeros_like(mask)
    frag_px = 0
    handled_defects = set()

    # Pass 1: paired defects sharing a tail tip vertex on the convex hull
    for i in range(len(defects)):
        if i in handled_defects:
            continue
        d1 = defects[i, 0]
        p_tip1 = c[int(d1[1])][0].astype(float)
        depth1 = float(d1[3]) / 256.0
        for j in range(len(defects)):
            if i == j or j in handled_defects:
                continue
            d2 = defects[j, 0]
            p_tip2 = c[int(d2[0])][0].astype(float)
            depth2 = float(d2[3]) / 256.0
            if np.linalg.norm(p_tip1 - p_tip2) <= max(16.0, 0.15 * dmax):
                if depth1 >= 4.0 and depth2 >= 4.0 and max(depth1, depth2) / dmax >= 0.16:
                    tip_pt = (p_tip1 + p_tip2) / 2.0
                    far1 = c[int(d1[2])][0].astype(float)
                    far2 = c[int(d2[2])][0].astype(float)

                    # Ensure tip protrudes farther from center than the notches
                    dtip = np.linalg.norm(tip_pt - centre)
                    if dtip <= np.linalg.norm(far1 - centre) or dtip <= np.linalg.norm(far2 - centre):
                        continue

                    v = far2 - far1
                    vn = float(np.linalg.norm(v))
                    if vn < 1e-6:
                        continue
                    n = np.array([-v[1], v[0]]) / vn
                    if np.dot(n, tip_pt - far1) < 0:
                        n = -n

                    side = n[0] * (xx - far1[0]) + n[1] * (yy - far1[1])
                    frag = (kept > 0) & (side > 0)
                    area = int(frag.sum())
                    if area == 0 or area > max_frag_frac * total:
                        continue
                    trial = kept & (frag == 0)
                    if containment(trial, text_xyxy) < min(min_cont, orig_cont):
                        continue
                    if not _narrows_to_tip(frag, far1, n):
                        continue  # a widening body wedge, not a tail
                    # a tail is much narrower than the balloon itself; a cut
                    # fragment as wide as the mask is a body piece
                    if _max_cross_width(frag, far1, n) > \
                            0.45 * _max_cross_width(kept, far1, n):
                        continue

                    kept = trial
                    fragments |= frag
                    frag_px += area
                    handled_defects.add(i)
                    handled_defects.add(j)

    # Pass 2: unpaired deep defects (e.g. leaked blob or single notch)
    for i in range(len(defects)):
        if i in handled_defects:
            continue
        d = defects[i, 0]
        depth = float(d[3]) / 256.0
        if depth < 6.0 or depth / dmax < thresh:
            continue
        p0 = c[int(d[0])][0].astype(float)
        p1 = c[int(d[1])][0].astype(float)
        far = c[int(d[2])][0].astype(float)

        dist0 = np.linalg.norm(p0 - centre)
        dist1 = np.linalg.norm(p1 - centre)
        tip_pt = p1 if dist1 > dist0 else p0
        dist_tip = max(dist0, dist1)
        dist_far = np.linalg.norm(far - centre)
        if dist_tip <= dist_far:
            continue

        n = tip_pt - far
        vn = float(np.linalg.norm(n))
        if vn < 1e-6:
            continue
        n /= vn

        side = n[0] * (xx - far[0]) + n[1] * (yy - far[1])
        frag = (kept > 0) & (side > 0)
        area = int(frag.sum())
        if area == 0 or area > max_frag_frac * total:
            continue
        trial = kept & (frag == 0)
        if containment(trial, text_xyxy) < min(min_cont, orig_cont):
            continue
        if not _narrows_to_tip(frag, far, n):
            continue  # a widening body wedge, not a tail
        if _max_cross_width(frag, far, n) > \
                0.45 * _max_cross_width(kept, far, n):
            continue  # a cut fragment as wide as the mask is a body piece

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


MERGE_COVER = 0.3  # neighbour-text containment above this means one merged blob


def split_merged(merged: np.ndarray, gray: np.ndarray, text_xyxy,
                 others) -> tuple[np.ndarray, np.ndarray | None]:
    """Split a merged white mask so only the selected text's balloon is kept.

    Adjacent balloons joined by a neck (or touching through a thin/broken
    wall) form a single white connected component, so the interior search
    returns all of them at once. A watershed with the text boxes as markers
    cuts the blob along the neck: the selected text keeps its basin, every
    neighbour keeps its own, ink walls outside the (dilated) text boxes form
    an excluded region the basins cannot cross.

    Returns ``(kept, cut)``: the selected basin clipped to ``merged``, and
    the seam band between text basins (``None`` when no neighbour marker
    landed inside the ROI). The seam has no ink by construction - pass its
    distance transform to ``measure_stroke_width`` as ``ignore_dt`` so the
    cut does not drag the outline coverage down.
    """
    H, W = merged.shape
    sel = tuple(int(v) for v in text_xyxy)
    near = []
    w = sel[2] - sel[0]
    h = sel[3] - sel[1]
    cx = (sel[0] + sel[2]) / 2
    cy = (sel[1] + sel[3]) / 2
    for o in others:
        o = tuple(int(v) for v in o)
        if o == sel:
            continue
        ox = (o[0] + o[2]) / 2
        oy = (o[1] + o[3]) / 2
        if abs(ox - cx) < 2 * w + 120 and abs(oy - cy) < 2 * h + 120:
            near.append(o)
    xs = [sel[0], sel[2]] + [o[0] for o in near] + [o[2] for o in near]
    ys = [sel[1], sel[3]] + [o[1] for o in near] + [o[3] for o in near]
    x1, y1 = max(0, min(xs) - 30), max(0, min(ys) - 30)
    x2, y2 = min(W, max(xs) + 30), min(H, max(ys) + 30)
    g = gray[y1:y2, x1:x2]
    markers = np.zeros((y2 - y1, x2 - x1), np.int32)

    def put(o, val):
        ox1, oy1 = max(0, o[0] - x1), max(0, o[1] - y1)
        ox2, oy2 = min(x2 - x1, o[2] - x1), min(y2 - y1, o[3] - y1)
        if ox2 > ox1 and oy2 > oy1 and not (markers[oy1:oy2, ox1:ox2] != 0).any():
            markers[oy1:oy2, ox1:ox2] = val
            return True
        return False

    put(sel, 1)
    n_mark = 1
    for i, o in enumerate(near):
        if put(o, i + 2):
            n_mark += 1
    if n_mark < 2:
        return merged, None
    boxes = cv2.dilate((markers != 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    ink = (g < DARK_STROKE).astype(np.uint8)  # thin walls: basin must abut the edge
    markers[(ink > 0) & (boxes == 0)] = 99
    cv2.watershed(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), markers)
    m1 = (markers == 1).astype(np.uint8)
    moth = (((markers >= 2) & (markers != 99)).astype(np.uint8))
    k5 = np.ones((5, 5), np.uint8)
    seam = (cv2.dilate(m1, k5) & cv2.dilate(moth, k5)).astype(bool)
    keep = np.zeros_like(merged)
    keep[y1:y2, x1:x2] = m1
    keep = keep & (merged > 0).astype(np.uint8)
    if not keep.any() or containment(keep.astype(np.uint8) * 255, sel) < 0.5:
        return merged, None  # split lost the text: keep the merged blob
    cut = np.zeros_like(merged)
    cut[y1:y2, x1:x2] = seam.astype(np.uint8) * 255
    return keep.astype(np.uint8) * 255, cut


def plausible_size(inner: np.ndarray, text_xyxy, bubble_xyxy) -> bool:
    """Reject page-fill masks: a balloon interior cannot dwarf its own box."""
    area = int((inner > 0).sum())
    if area == 0:
        return False
    tx1, ty1, tx2, ty2 = (int(v) for v in text_xyxy)
    t_area = max(1, (tx2 - tx1) * (ty2 - ty1))
    if area > MAX_INNER_FRAC * t_area:
        return False
    if bubble_xyxy is not None and not np.array_equal(
            np.asarray(bubble_xyxy, dtype=int), np.asarray(text_xyxy, dtype=int)):
        bx1, by1, bx2, by2 = (int(v) for v in bubble_xyxy)
        b_area = max(1, (bx2 - bx1) * (by2 - by1))
        if area > MAX_INNER_VS_BOX * b_area:
            return False
    return True


def has_bubble_box(text_xyxy, bubble_xyxy) -> bool:
    """True when the detector/user gave a box enclosing the text.

    This is the strongest balloon prior the pipeline holds. Text printed
    directly on the page background (credits, captions, signs, onomatopoeia)
    is classified ``text_free`` and carries no bubble box - its light
    surroundings are geometrically indistinguishable from an enclosed balloon,
    so the only safe answer is to refuse instead of flooding the panel white.
    A box equal to the text box is accepted: the detector emits that for small
    SFX balloons, and it is what a user-drawn rectangle produces.
    """
    if bubble_xyxy is None:
        return False
    t = np.asarray(text_xyxy, dtype=int)
    b = np.asarray(bubble_xyxy, dtype=int)
    slack = 3
    return (b[0] - slack <= t[0] and b[1] - slack <= t[1]
            and b[2] + slack >= t[2] and b[3] + slack >= t[3])


def estimate_fill(rgb: np.ndarray, inner: np.ndarray,
                  gray: np.ndarray) -> tuple[int, int, int]:
    """Flat colour of the balloon interior (median of its light pixels).

    A bubble is repainted with a flat fill; using a hard-coded white destroyed
    cream / yellow / red balloons. The median is taken over interior pixels
    that pass the interior grey threshold, so glyph pixels do not drag it
    towards black. Very light interiors snap to pure white so white balloons
    keep a clean plate instead of a faintly grey one.
    """
    m = (inner > 0) & (gray >= WHITE)
    px = rgb[m]
    if px.shape[0] < 20:
        return (255, 255, 255)
    med = np.median(px, axis=0)
    if float(med.min()) >= NEAR_WHITE:
        return (255, 255, 255)
    return tuple(int(round(float(v))) for v in med)


def recover_bubble(rgb: np.ndarray, text_xyxy, bubble_xyxy=None,
                   text_mask=None, other_texts=None) -> dict | None:
    """Recommended path: C -> Cx, falling back to C2 -> C2x.

    Returns ``{"interior", "outer", "fragments", "stroke_w", "method", "fill",
    "trim", "split"}`` or ``None`` when no plausible bubble was found.
    ``stroke_w`` is the measured outline width in px; ``outer`` is the body
    mask after tail/fragment trimming, ``fragments`` marks what was cut,
    ``fill`` is the interior colour, ``split`` tells whether a merged blob
    was cut away from a neighbouring balloon.

    ``other_texts`` lists the remaining text boxes on the page: when the
    white component around this text also covers a neighbour (adjacent
    balloons joined by a neck), the blob is watershed-split and only this
    text's basin is kept.

    Two guards separate a balloon from the page background before any repaint:
    the interior must be plausibly sized (``plausible_size``) and must sit on an
    ink outline (``measure_stroke_width`` >= ``MIN_STROKE``). Without them a
    white page margin / caption strip was classified as a bubble and the
    expander flooded the panel with white.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    others = [tuple(int(v) for v in o) for o in (other_texts or [])]
    roi = (_union_roi(text_xyxy, others, gray.shape, bubble_xyxy=bubble_xyxy)
           if others else None)
    inner = interior_white(gray, text_xyxy, bubble_xyxy, roi)
    if inner is not None and containment(inner, text_xyxy) < 0.5:
        inner = None  # the white blob found is not the bubble around this text

    if inner is not None:
        method = "C->Cx"
    else:
        inner = interior_color(rgb, gray, text_xyxy, bubble_xyxy, text_mask, roi)
        if inner is not None and containment(inner, text_xyxy) < 0.5:
            inner = None
        method = "C2->C2x"
    if inner is None:
        return None
    if not has_bubble_box(text_xyxy, bubble_xyxy):
        return None  # no balloon prior: page background / open art

    split = False
    ignore_dt = None
    sel_box = tuple(int(v) for v in text_xyxy)
    if others and any(containment(inner, o) > MERGE_COVER
                     for o in others if tuple(int(v) for v in o) != sel_box):
        inner, cut = split_merged(inner, gray, text_xyxy, others)
        if cut is not None:
            split = True
            ignore_dt = cv2.distanceTransform(
                (cut == 0).astype(np.uint8), cv2.DIST_L2, 5)
    if not plausible_size(inner, text_xyxy, bubble_xyxy):
        return None

    stroke_w = measure_stroke_width(gray, inner, ignore_dt=ignore_dt)
    if stroke_w < MIN_STROKE:
        # no ink outline -> page background / screentone, not a balloon
        return None
    outer = expand_to_outer(inner, stroke_w, gray=gray)
    if not outer.any():
        return None
    outer, fragments, trim_stats = trim_fragments(outer, text_xyxy)
    return {
        "interior": inner,
        "outer": outer,
        "fragments": fragments,
        "stroke_w": stroke_w,
        "method": method,
        "fill": estimate_fill(rgb, inner, gray),
        "trim": trim_stats,
        "split": split,
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
    return smooth_mask(fill_holes(expanded))


def repaint(rgb: np.ndarray, expanded: np.ndarray, stroke_r: int,
            fill: tuple[int, int, int] = (255, 255, 255),
            sigma: float = 0.8) -> np.ndarray:
    """Flat fill + outline ring at exactly ``stroke_r`` px (variant B from Stage 7).

    ``fill`` is the interior colour from ``estimate_fill`` (defaults to white),
    so cream / tinted balloons keep their tone instead of being flattened white.

    The ring is ``stroke_r`` deep from the mask edge: a distance transform of
    the mask gives every interior pixel its depth from the boundary, and the
    ring alpha ramps 1 -> 0 across the outermost pixel
    (``clip(stroke_r + 1 - dist)``). The dark core (alpha > 0.5) is therefore
    exactly ``stroke_r`` px wide everywhere - the supersampled
    ``expanded - erode(expanded, r)`` band it replaces picked up a sub-pixel
    phase of the boundary through INTER_AREA and rendered 3-7 px for a 4 px
    stroke. The blurred tail of the ring is multiplied by the softly blurred
    mask, so the alpha fades to zero at the smoothed mask boundary: nothing
    bleeds in from the background and no fill-coloured halo is painted over
    it. A slight blur keeps the edge anti-aliased and comic-crisp.
    """
    out = rgb.copy()
    m = (expanded > 0)
    if not m.any():
        return out
    f = tuple(int(np.clip(v, 0, 255)) for v in fill)
    r = int(stroke_r)
    if r <= 0:
        out[m] = f
        return out  # no dark outline on this bubble: repaint the fill only

    x1, y1, x2, y2 = mask_bbox(expanded, pad=r + 6, shape=rgb.shape[:2])
    sub = out[y1:y2, x1:x2]
    subm = m[y1:y2, x1:x2]
    sub[subm] = f

    dist = cv2.distanceTransform(subm.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(r + 1.0 - dist, 0.0, 1.0).astype(np.float32)
    alpha[~subm] = 0.0  # dist is 0 outside the mask: keep the ring inside it
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)

    # The blurred ring still leaks a tail ~1px outside the mask that darkens
    # light background art. Multiply by the mask blurred with a NARROW
    # gaussian (0.5): the tail is suppressed to ~0.03 (invisible) while the
    # fringe inside stays anti-aliased - no bleed-in, no fill-coloured halo.
    m_soft = cv2.GaussianBlur(subm.astype(np.float32), (0, 0), 0.5)
    alpha = alpha * m_soft

    a = alpha[..., None]
    sub[:] = (sub * (1 - a)).astype(np.uint8)
    return out


def repaint_transplant(rgb: np.ndarray, outer: np.ndarray, expanded: np.ndarray,
                       px: int, fill: tuple[int, int, int] = (255, 255, 255),
                       wmax: int = 6, aa_width: float = 1.4,
                       fragments: np.ndarray | None = None) -> np.ndarray:
    """Flat fill + original outline transplanted outward by ``px`` (variant A).

    ``outer`` is the pre-expansion body mask, ``expanded`` the dilated mask.
    The whole expanded interior is filled with ``fill`` (this erases the old
    outline, which would otherwise stay behind as a ghost ring, plus the
    source glyphs so translated text lands on a clean plate). The new edge
    ring (``wmax`` px deep from the expanded boundary) is then copied from
    the true original edge strip: every new-edge pixel at depth ``d`` takes
    the colour of the original pixel at the same depth ``d`` from the
    ``outer`` boundary, i.e. shifted ``px`` inward along the boundary normal
    and bilinearly sampled. Width, ink colour and anti-aliasing therefore
    match the original balloon instead of a uniform synthetic band.

    When ``fragments`` (tail mask / adjoining bubble cuts) is provided,
    the outline alpha is suppressed across portals so lines are never drawn
    across open tails or corridors between merged balloons.

    The outer edge of the ring is alpha-blended with the background over
    ``aa_width`` px (default 1.4) so the boundary fades smoothly instead of
    producing a hard staircase from the binary mask.
    """
    out = rgb.copy()
    subm = (expanded > 0).astype(np.uint8)
    if not subm.any():
        return out
    f = tuple(int(np.clip(v, 0, 255)) for v in fill)
    out[subm > 0] = f
    base = (outer > 0).astype(np.uint8)
    new_band = (subm > 0) & (base == 0)
    if not new_band.any() or px <= 0:
        return out

    # Base distance transform of the pre-expansion mask (base)
    d_out_base = cv2.distanceTransform((base == 0).astype(np.uint8), cv2.DIST_L2, 5)
    d_in_base = cv2.distanceTransform(base, cv2.DIST_L2, 5)
    sdf_old = (d_out_base - d_in_base).astype(np.float32)

    dist = cv2.distanceTransform(subm, cv2.DIST_L2, 5)
    # Smooth distance field with a gentle gaussian for stable, noise-free normals
    dist_smooth = cv2.GaussianBlur(dist, (3, 3), 0.8)
    gx = cv2.Sobel(dist_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist_smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag[mag < 1e-6] = 1.0
    nx, ny = gx / mag, gy / mag  # inward normals (dist grows inward)
    H, W = subm.shape
    yy, xx = np.mgrid[0:H, 0:W]
    # Widen the ring zone by aa_width so the blend tail has room to fade
    ring = new_band & (dist <= float(wmax) + 1.0 + aa_width)
    if not ring.any():
        return out
    rx, ry = xx[ring], yy[ring]

    # Target depth in the old stroke corresponds to current depth from the expanded boundary
    # Newton iteration: ensures sx, sy lands on the exact same SDF depth in the source bubble,
    # completely eliminating thinning or gaps caused by curvature differences or local variations.
    target_v = dist[ring] - float(px)
    sx = (rx + px * nx[ring]).astype(np.float32)
    sy = (ry + px * ny[ring]).astype(np.float32)

    # 1 iteration of Newton refinement on the source distance field
    sx_grid = np.zeros((H, W), dtype=np.float32)
    sy_grid = np.zeros((H, W), dtype=np.float32)
    sx_grid[ring] = sx
    sy_grid[ring] = sy
    sampled_sdf = cv2.remap(sdf_old, sx_grid, sy_grid, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    residual = sampled_sdf[ring] - target_v
    # Cap residual step to prevent overshooting near sharp corners
    residual = np.clip(residual, -2.0, 2.0)
    sx = np.clip(sx - residual * nx[ring], 0.0, W - 1.0)
    sy = np.clip(sy - residual * ny[ring], 0.0, H - 1.0)

    src = rgb.astype(np.float32)

    def _sample(cx, cy):
        x0 = np.clip(np.floor(cx).astype(int), 0, W - 1)
        y0 = np.clip(np.floor(cy).astype(int), 0, H - 1)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)
        fx = (cx - x0).astype(np.float32)
        fy = (cy - y0).astype(np.float32)
        return (src[y0, x0] * (1 - fx[:, None]) * (1 - fy[:, None])
                + src[y0, x1] * fx[:, None] * (1 - fy[:, None])
                + src[y1, x0] * (1 - fx[:, None]) * fy[:, None]
                + src[y1, x1] * fx[:, None] * fy[:, None])

    samp_c = _sample(sx, sy)
    # Check a subpixel +-0.8px step along the normal to lock onto the true ink core
    # even when discrete dilation slightly misaligns the normal phase.
    samp_in = _sample(sx + 0.8 * nx[ring], sy + 0.8 * ny[ring])
    samp_out = _sample(sx - 0.8 * nx[ring], sy - 0.8 * ny[ring])

    lum_c = samp_c[:, 0] * 0.299 + samp_c[:, 1] * 0.587 + samp_c[:, 2] * 0.114
    lum_i = samp_in[:, 0] * 0.299 + samp_in[:, 1] * 0.587 + samp_in[:, 2] * 0.114
    lum_o = samp_out[:, 0] * 0.299 + samp_out[:, 1] * 0.587 + samp_out[:, 2] * 0.114

    min_lum = np.minimum(lum_c, np.minimum(lum_i, lum_o))
    samp = samp_c.copy()
    pick_i = (lum_i == min_lum) & (lum_i < lum_c - 12)
    pick_o = (lum_o == min_lum) & (lum_o < lum_c - 12)
    samp[pick_i] = samp_in[pick_i]
    samp[pick_o] = samp_out[pick_o]

    # Alpha ramp: dist ≥ aa_width → 1.0 (solid), dist → 0 fades to 0.
    # This anti-aliases the outer edge of the ring over ~aa_width px against the
    # true original page background (not the repainted fill), avoiding a light halo.
    edge_dist = dist[ring]
    alpha = np.clip(edge_dist / aa_width, 0.0, 1.0).astype(np.float32)

    # Portal / tail cut suppression: do not paint outline across open cuts or tails
    if fragments is not None and (fragments > 0).any():
        portal_dist = cv2.distanceTransform((fragments == 0).astype(np.uint8), cv2.DIST_L2, 5)
        portal_suppress = np.clip((portal_dist[ring] - 1.0) / 4.0, 0.0, 1.0).astype(np.float32)
        alpha = alpha * portal_suppress

    bg = rgb[ring].astype(np.float32)
    out[ring] = np.round(samp * alpha[:, None] + bg * (1.0 - alpha[:, None])).astype(np.uint8)
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
