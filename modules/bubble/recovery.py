"""Experimental manual bubble recovery + expansion (no ML, no new deps).

Ports the validated recipe from ``scratch/bubble_research`` (REPORT.md):

recovery   interior mask via C (white-threshold connected components inside the
           detector bubble box), falling back to C2 (colour-adaptive) when the
           text is not contained, then expanding through the stroke to the
           outer contour (Cx / C2x) with an auto-estimated stroke width.
expansion  dilate the outer mask, repaint a flat fill and redraw the outline as
           a distance-transform ring (``stroke_r`` deep from the mask edge).
           The ring hits the requested width exactly and stays constant along
           the perimeter, unlike ``cv2.drawContours`` which quantises stroke
           width in ~2px steps (Stage 7 calibration).

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


def interior_white(gray: np.ndarray, text_xyxy, bubble_xyxy) -> np.ndarray | None:
    """C: threshold + connected components inside the ROI around the text."""
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
                   text_mask) -> np.ndarray | None:
    """C2: colour-adaptive interior (white / cream / yellow bubbles)."""
    fill = _fill_color(rgb, text_xyxy, bubble_xyxy, text_mask)
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
                    max_r: int = MAX_STROKE) -> dict:
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
    hits = []
    clean = []
    blocked = 0
    ink_contact = 0
    for i in range(n):
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
    prof["n"] = n
    prof["runs"] = hits
    prof["clean"] = np.asarray(clean, dtype=bool)
    prof["blocked"] = blocked
    prof["coverage"] = float(ink_contact) / n if n else 0.0
    if hits.size:
        prof["med"] = float(np.median(hits))
        prof["p25"] = float(np.percentile(hits, 25))
    return prof


def measure_stroke_width(gray: np.ndarray, interior: np.ndarray,
                         max_r: int = MAX_STROKE) -> int:
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
    prof = _stroke_profile(gray, interior, max_r)
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


def expand_to_outer(interior: np.ndarray, k: int) -> np.ndarray:
    cur = (interior > 0).astype(np.uint8)
    for _ in range(max(0, k)):
        cur = cv2.dilate(cur, ELL3)
    return fill_holes(cur.astype(np.uint8) * 255)


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
                   text_mask=None) -> dict | None:
    """Recommended path: C -> Cx, falling back to C2 -> C2x.

    Returns ``{"interior", "outer", "fragments", "stroke_w", "method", "fill",
    "trim"}`` or ``None`` when no plausible bubble was found. ``stroke_w`` is the
    measured outline width in px; ``outer`` is the body mask after tail/fragment
    trimming, ``fragments`` marks what was cut, ``fill`` is the interior colour.

    Two guards separate a balloon from the page background before any repaint:
    the interior must be plausibly sized (``plausible_size``) and must sit on an
    ink outline (``measure_stroke_width`` >= ``MIN_STROKE``). Without them a
    white page margin / caption strip was classified as a bubble and the
    expander flooded the panel with white.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    inner = interior_white(gray, text_xyxy, bubble_xyxy)
    if inner is not None and containment(inner, text_xyxy) < 0.5:
        inner = None  # the white blob found is not the bubble around this text

    if inner is not None:
        method = "C->Cx"
    else:
        inner = interior_color(rgb, gray, text_xyxy, bubble_xyxy, text_mask)
        if inner is not None and containment(inner, text_xyxy) < 0.5:
            inner = None
        method = "C2->C2x"
    if inner is None:
        return None
    if not has_bubble_box(text_xyxy, bubble_xyxy):
        return None  # no balloon prior: page background / open art
    if not plausible_size(inner, text_xyxy, bubble_xyxy):
        return None

    stroke_w = measure_stroke_width(gray, inner)
    if stroke_w < MIN_STROKE:
        # no ink outline -> page background / screentone, not a balloon
        return None
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
        "fill": estimate_fill(rgb, inner, gray),
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
    stroke. A slight blur keeps the edge anti-aliased and comic-crisp.
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
