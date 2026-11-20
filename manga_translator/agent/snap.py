"""Snap an approximate box to the text strokes inside it.

The model is good at saying *where* text is but its boxes are loose (M0.5:
IoU median 0.79, sometimes truncated). This module turns such a box into what
a region needs (design doc, "精準操作設計": 吸附):

* a tight box and one quad per column (vertical) or row (horizontal);
* the font size, text and background colors;
* the pixels to erase (ink, slightly dilated) for the inpainting mask.

Only strokes that start inside the given box are seeds; nearby strokes of a
similar size are then added, so text the box cut off is still captured, while
bubble outlines, panel borders and table lines (long strokes) are left out.
Text that already belongs to another region is excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

MAX_ELONGATION = 18.0
"""A stroke longer than this many times its thickness is a line (bubble
outline, panel border, table rule), not text."""


@dataclass
class SnapResult:
    bbox: Tuple[int, int, int, int]
    """Tight box around the kept strokes, page pixels."""
    lines: List[List[List[float]]]
    """One quad per column (vertical) or row (horizontal), page pixels."""
    font_size: int
    fg: List[int]
    bg: List[int]
    erase: np.ndarray
    """0/1 patch of pixels to erase, placed at ``erase_origin``."""
    erase_origin: Tuple[int, int]
    strokes: int
    warning: str = ""


def _runs(profile: np.ndarray, min_gap: int) -> List[Tuple[int, int]]:
    """Runs of True in ``profile``; gaps shorter than ``min_gap`` are bridged."""
    runs: List[List[int]] = []
    start = None
    for i, on in enumerate(profile):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(profile)])
    merged: List[List[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < min_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    return [(a, b) for a, b in merged]


def _gap(a: Sequence[int], b: Sequence[int]) -> int:
    """Chebyshev gap between two boxes (x1, y1, x2, y2); 0 when they touch."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


def snap_text(image: np.ndarray, box: Sequence[float], direction: str,
              exclude: Optional[np.ndarray] = None, margin: Optional[int] = None) -> Optional[SnapResult]:
    """Snap ``box`` (x1, y1, x2, y2, page pixels, loose) to the text in ``image`` (RGB).

    ``exclude`` is a page-sized 0/1 array of pixels that belong to other
    regions. Returns None when the box holds no text-like strokes.
    """
    height, width = image.shape[:2]
    x1, x2 = sorted((int(round(box[0])), int(round(box[2]))))
    y1, y2 = sorted((int(round(box[1])), int(round(box[3]))))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    m = margin if margin is not None else int(max(16, min(96, 0.5 * max(x2 - x1, y2 - y1))))
    cx1, cy1 = max(0, x1 - m), max(0, y1 - m)
    cx2, cy2 = min(width, x2 + m), min(height, y2 + m)
    crop = image[cy1:cy2, cx1:cx2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.copy()
    ix1, iy1, ix2, iy2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
    inner = gray[iy1:iy2, ix1:ix2]
    if float(inner.std()) < 8.0:  # flat area: nothing written here
        return None
    threshold, _ = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray < threshold
    # Text is the minority inside a box drawn around it; white text on a dark
    # background flips the polarity.
    ink = ~dark if dark[iy1:iy2, ix1:ix2].mean() > 0.5 else dark
    ink = ink.astype(np.uint8)
    if exclude is not None:
        ink[exclude[cy1:cy2, cx1:cx2] > 0] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # Stroke thickness per component: text strokes are short for their
    # thickness, bubble outlines and panel borders are long thin curves.
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    thickness = np.zeros(count, np.float32)
    np.maximum.at(thickness, labels.ravel(), dist.ravel())
    ch, cw = ink.shape
    comps = []
    for label in range(1, count):
        sx, sy, sw, sh, area = (int(v) for v in stats[label])
        if area < 3:
            continue
        size = max(sw, sh)
        comps.append({"label": label, "box": (sx, sy, sx + sw, sy + sh), "size": size,
                      "touches": sx == 0 or sy == 0 or sx + sw >= cw or sy + sh >= ch,
                      "textlike": size <= MAX_ELONGATION * max(1.0, 2.0 * float(thickness[label]))})

    def inside_inner(c) -> bool:
        bx1, by1, bx2, by2 = c["box"]
        mx, my = (bx1 + bx2) / 2, (by1 + by2) / 2
        return ix1 <= mx <= ix2 and iy1 <= my <= iy2

    seeds = [c for c in comps if inside_inner(c) and not c["touches"] and c["textlike"]]
    if not seeds:
        return None
    sizes = [c["size"] for c in seeds]
    # Ellipsis dots and furigana are many and small: estimate the character
    # size from the large strokes too, or the kanji next to them look oversized.
    char = float(np.clip(max(np.percentile(sizes, 75), 0.5 * max(sizes)), 6, 200))
    seeds = [c for c in seeds if c["size"] <= 2.5 * char]
    if not seeds:
        return None
    # Keep only strokes on the same background as the large text strokes
    # (usually the inside of the bubble). Screentone outside the bubble is
    # cut off by the bubble outline even when the box reaches past it.
    bg = (ink == 0).astype(np.uint8)
    _n, bg_labels = cv2.connectedComponents(bg, connectivity=4)
    big = np.isin(labels, [c["label"] for c in seeds if c["size"] >= 0.5 * char]).astype(np.uint8)
    touched = bg_labels[(cv2.dilate(big, np.ones((3, 3), np.uint8)) > 0) & (bg > 0)]
    touched = touched[touched > 0]
    if touched.size:
        main_bg = (bg_labels == int(np.bincount(touched).argmax())).astype(np.uint8)
        ring = (cv2.dilate(main_bg, np.ones((3, 3), np.uint8)) > 0) & (ink > 0)
        on_main = set(np.unique(labels[ring]).tolist())
        comps = [c for c in comps if c["label"] in on_main]
        seeds = [c for c in seeds if c["label"] in on_main] or seeds
    kept = {c["label"]: c for c in seeds}
    union = [min(c["box"][0] for c in seeds), min(c["box"][1] for c in seeds),
             max(c["box"][2] for c in seeds), max(c["box"][3] for c in seeds)]
    reach = max(3.0, 0.6 * char)
    for _ in range(12):
        grew = False
        for c in comps:
            if (c["label"] in kept or c["touches"] or not c["textlike"] or c["size"] > 2 * char
                    or _gap(c["box"], union) > reach):
                continue
            kept[c["label"]] = c
            union = [min(union[0], c["box"][0]), min(union[1], c["box"][1]),
                     max(union[2], c["box"][2]), max(union[3], c["box"][3])]
            grew = True
        if not grew:
            break

    keep_mask = np.isin(labels, list(kept)).astype(np.uint8)
    ux1, uy1, ux2, uy2 = union
    sub = keep_mask[uy1:uy2, ux1:ux2]
    vertical = direction.startswith("v")
    profile = sub.any(axis=0 if vertical else 1)
    runs = _runs(profile, max(2, int(round(0.15 * char))))
    spans = [b - a for a, b in runs]
    widest = max(spans) if spans else 0
    main = [(a, b) for a, b in runs if b - a >= 0.6 * widest] or runs
    lines = []
    for a, b in main:
        part = sub[:, a:b] if vertical else sub[a:b, :]
        other = np.flatnonzero(part.any(axis=1 if vertical else 0))
        if not len(other):
            continue
        o1, o2 = int(other[0]), int(other[-1]) + 1
        if vertical:
            qx1, qy1, qx2, qy2 = ux1 + a, uy1 + o1, ux1 + b, uy1 + o2
        else:
            qx1, qy1, qx2, qy2 = ux1 + o1, uy1 + a, ux1 + o2, uy1 + b
        qx1, qx2, qy1, qy2 = qx1 + cx1, qx2 + cx1, qy1 + cy1, qy2 + cy1
        lines.append([[float(qx1), float(qy1)], [float(qx2), float(qy1)], [float(qx2), float(qy2)], [float(qx1), float(qy2)]])
    if not lines:
        return None
    font_size = int(round(float(np.median([b - a for a, b in main]))))

    rgb = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    fg = np.median(rgb[keep_mask > 0], axis=0)
    ring = np.zeros_like(keep_mask)
    ring[max(0, uy1 - 3):uy2 + 3, max(0, ux1 - 3):ux2 + 3] = 1
    ring[cv2.dilate(keep_mask, np.ones((3, 3), np.uint8)) > 0] = 0
    bg = np.median(rgb[ring > 0], axis=0) if ring.any() else np.array([255, 255, 255])

    radius = max(2, int(round(0.3 * char)))
    erase = cv2.dilate(keep_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    ys, xs = np.nonzero(erase)
    ey1, ey2, ex1, ex2 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1

    bbox = (ux1 + cx1, uy1 + cy1, ux2 + cx1, uy2 + cy1)
    given = (x2 - x1) * (y2 - y1)
    found = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    warning = ""
    if found > 3 * given:
        warning = "the strokes found reach far outside the box you gave"
    elif found < 0.1 * given:
        warning = "strokes were found only in a small part of the box you gave"
    return SnapResult(bbox=bbox, lines=lines, font_size=max(6, font_size),
                      fg=[int(v) for v in fg], bg=[int(v) for v in bg],
                      erase=erase[ey1:ey2, ex1:ex2].copy(), erase_origin=(cx1 + ex1, cy1 + ey1),
                      strokes=len(kept), warning=warning)
