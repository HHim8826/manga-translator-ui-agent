"""Snap an approximate box to the text strokes inside it.

The model is good at saying *where* text is but its boxes are loose (M0.5:
IoU median 0.79, sometimes truncated). This module turns such a box into what
a region needs (snapping):

* a tight box and one quad per column (vertical) or row (horizontal);
* the font size, text and background colors;
* the pixels to erase (ink, slightly dilated) for the inpainting mask.

Only strokes that start inside the given box are seeds; nearby strokes of a
similar size are then added, so text the box cut off is still captured, while
bubble outlines, panel borders and table lines (long strokes) are left out.
Text that already belongs to another region is excluded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

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
    angle: float = 0.0
    """Slant of the line (``TextBlock.angle``): the angle of its direction on the page in degrees,
    y down, so a line rising to the right is negative; 0 for text that runs along the axes."""


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


def _split_touching(runs: List[Tuple[int, int]], sub: np.ndarray, vertical: bool,
                    char: float) -> List[Tuple[int, int]]:
    """Cut the ``runs`` of the profile of ``sub`` (0/1 ink) as long as two lines or more at the
    clear valleys of ink between them: lines of handwriting set so tight that they touch (ch42 p1).
    A valley holds at most 30% of the run's usual ink, and every line cut off looks like one: at
    least 0.6 ``char`` across, 1.2 ``char`` along and a fifth of the ink of the heaviest (a big
    sound effect of a few thin strokes is not cut)."""
    ink = sub.sum(axis=0 if vertical else 1)

    def line_like(parts: List[Tuple[int, int]]) -> bool:
        heaviest = max(int(ink[p:q].sum()) for p, q in parts)
        for p, q in parts:
            span = np.flatnonzero((sub[:, p:q] if vertical else sub[p:q, :]).any(axis=1 if vertical else 0))
            if (q - p < 0.6 * char or not len(span) or span[-1] - span[0] + 1 < 1.2 * char
                    or ink[p:q].sum() < 0.2 * heaviest):
                return False
        return True

    out: List[Tuple[int, int]] = []
    for a, b in runs:
        length = b - a
        count = int(round(length / max(char, 1.0)))
        if length < 1.8 * char or count < 2:
            out.append((a, b))
            continue
        seg = ink[a:b].astype(np.float64)
        usual = float(np.median(seg[seg > 0])) if (seg > 0).any() else 0.0
        cuts = []
        for k in range(1, count):
            at, slack = k * length / count, 0.3 * length / count
            lo, hi = max(1, int(at - slack)), min(length - 1, int(at + slack) + 1)
            if hi > lo:
                row = lo + int(np.argmin(seg[lo:hi]))
                if seg[row] <= 0.3 * usual and (not cuts or row > cuts[-1]):
                    cuts.append(row)
        edges = [0] + cuts + [length]
        parts = [(a + edges[i], a + edges[i + 1]) for i in range(len(edges) - 1)]
        if cuts and line_like(parts):
            out.extend(parts)
        else:
            out.append((a, b))
    return out


def _gap(a: Sequence[int], b: Sequence[int]) -> int:
    """Chebyshev gap between two boxes (x1, y1, x2, y2); 0 when they touch."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


SLANT_MIN, SLANT_MAX = 5.0, 45.0
SLANT_INK = 0.8
"""A line of text tilted by this much (degrees) gets a slanted quad: a label written along a road or
a roof on a map. Less is lettering that runs along the axes; more is read in the other direction."""


def _bracket(box: Sequence[int], char: float, vertical: bool) -> bool:
    """A stroke as long across the lines as more than one of them and thin along them: a bracket
    around a text of more lines."""
    width, height = box[2] - box[0], box[3] - box[1]
    across, along = (width, height) if vertical else (height, width)
    return across >= 1.5 * char and across >= 3 * along


def _tone(small: list, strokes: list, dot: float, largest: float) -> set:
    """The labels of the ``small`` components that are screentone around the ``strokes`` of a text.
    A box is on tone when some of its dots lie in a regular lattice: their 4 nearest neighbours
    about as far away as each other and about their size (even in a small box cut by the white rim
    around lettering and with two tones in it, a few do; the small strokes of handwriting, a column
    of kana and dakuten, do not). Then a dot with 3 or more others close by is tone (the two marks
    of a dakuten and the middle dot of an ellipsis have fewer); so are the dots hanging on to such a
    lattice (cut in half by the white rim) that do not touch a stroke, and the dots away from the
    strokes."""
    if len(small) < 5:
        return set()
    mids = np.array([((c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2) for c in small])
    sizes = np.array([c["size"] for c in small], np.float64)
    tree = cKDTree(mids)
    dist, idx = tree.query(mids, k=5)
    even = dist[:, 4] <= 1.6 * dist[:, 1]
    around = sizes[idx[:, 1:]]
    like = np.all((around >= 0.5 * sizes[:, None]) & (around <= 2.0 * sizes[:, None]), axis=1)
    if (even & like).sum() < max(8, 0.1 * len(small)):
        return set()
    reach = 3.0 * np.maximum(sizes, max(dot, 2.0))
    crowd = np.array([len(n) - 1 for n in tree.query_ball_point(mids, r=reach)])
    lattice = crowd >= 3
    edge = np.zeros(len(small), bool)
    if lattice.any():
        pitch = float(np.median(dist[lattice, 1]))
        pairs = tree.query_pairs(1.5 * pitch, output_type="ndarray").reshape(-1, 2)
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(small),) * 2)
        _count, group = connected_components(graph, directed=False)
        edge = np.isin(group, np.unique(group[lattice])) & ~lattice
    near = max(2.0, 0.3 * largest)
    touch = max(2.0, 0.1 * largest)
    tone = set()
    for c, in_lattice, on_edge in zip(small, lattice, edge):
        gap = min(_gap(c["box"], t["box"]) for t in strokes)
        if in_lattice or gap > near or (on_edge and gap > touch):
            tone.add(c["label"])
    return tone


def _slanted(labels: np.ndarray, ids: Sequence[int], char: float, vertical: bool):
    """One slanted line of text among the strokes ``ids`` of ``labels`` (connected components): its
    quad (``labels`` pixels), its size across the line, its slant (``TextBlock.angle``), the strokes
    on it and its axis; None when the strokes run along the axes. The slant is the one at which the
    ink, seen across the line, is most heaped up (the sum of squares of its profile, as in
    deskewing a scanned page; in steps of an eighth of a character, so it follows the glyphs and
    not the strokes in them: the many level strokes of a kanji heap up at 0°). An upright box around a slanted line also takes in the corners of the
    labels next to it (a roof label under a road label on a map): the line keeps the strokes in the
    narrowest band that holds most of the ink (SLANT_INK of it), the rest are left to their labels.
    The quad must be clearly smaller than the upright box, so a short word of uneven glyphs stays
    upright."""
    ids = list(ids)
    ys, xs = np.nonzero(np.isin(labels, ids))
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64) + 0.5
    sample = pts[:: max(1, len(pts) // 4000)]
    count = len(sample)
    k = max(1, int(SLANT_INK * count)) - 1

    def across_axis(tilt: float) -> np.ndarray:
        t = math.radians(tilt)
        return np.array([math.cos(t), math.sin(t)]) if vertical else np.array([-math.sin(t), math.cos(t)])

    def heaped(tilt: float) -> float:
        place = sample @ across_axis(tilt)
        place = (place - place.min()) / step
        cell = np.floor(place).astype(int)
        part = place - cell
        profile = (np.bincount(cell, 1.0 - part, cell.max() + 2) + np.bincount(cell + 1, part, cell.max() + 2))
        profile = np.convolve(profile, [0.25, 0.5, 0.25])
        return float((profile ** 2).sum())

    step = max(1.0, char / 8.0)
    scores = {float(tilt): heaped(tilt) for tilt in np.arange(-SLANT_MAX, SLANT_MAX + 0.25, 0.5)}
    tilt = max(scores, key=lambda a: (scores[a], -abs(a)))
    if abs(tilt) < SLANT_MIN or scores[tilt] < 1.15 * scores[0.0]:  # upright reads about as well
        return None
    spread = np.sort(sample @ across_axis(tilt))
    widths = spread[k:] - spread[:count - k]
    i = int(np.argmin(widths))
    width, low, high = float(widths[i]), float(spread[i]), float(spread[i + k])
    axis_across = across_axis(tilt)
    slack = 0.25 * width
    on_line, glyphs = [], 0
    for i in ids:
        where = np.argwhere(labels == i)
        cy, cx = where.mean(axis=0) + 0.5
        if low - slack <= cx * axis_across[0] + cy * axis_across[1] <= high + slack:
            on_line.append(i)
            glyphs += int((where.max(axis=0) - where.min(axis=0)).max() + 1 >= 0.5 * char)
    if glyphs < 2:  # one stroke and a few dots: no line of glyphs to follow
        return None
    ys, xs = np.nonzero(np.isin(labels, on_line))
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64) + 0.5
    evals = np.linalg.eigvalsh(np.cov((pts - pts.mean(axis=0)).T))
    if evals[1] < 2.5 * max(evals[0], 1e-6):  # not a line: about as long as it is wide
        return None
    t = math.radians(tilt)
    u = np.array([math.cos(t), math.sin(t)])  # along a row, across a column
    v = np.array([-math.sin(t), math.cos(t)])  # across a row, along a column
    pu, pv = pts @ u, pts @ v
    a1, a2, b1, b2 = pu.min() - 0.5, pu.max() + 0.5, pv.min() - 0.5, pv.max() + 0.5
    upright = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
    if (a2 - a1) * (b2 - b1) > 0.85 * upright:
        return None
    quad = [a1 * u + b1 * v, a2 * u + b1 * v, a2 * u + b2 * v, a1 * u + b2 * v]
    across = (a2 - a1) if vertical else (b2 - b1)
    axis = (pts.mean(axis=0), axis_across, 0.75 * width)  # a point on the line, its normal, the reach off it
    return quad, across, tilt, on_line, axis


def snap_text(image: np.ndarray, box: Sequence[float], direction: str,
              exclude: Optional[np.ndarray] = None, margin: Optional[int] = None,
              grow: bool = True) -> Optional[SnapResult]:
    """Snap ``box`` (x1, y1, x2, y2, page pixels, loose) to the text in ``image`` (RGB).

    ``exclude`` is a page-sized 0/1 array of pixels that belong to other
    regions. With ``grow`` off only the strokes whose middle lies inside the
    box are taken (cutting part of a detected line: the strokes next to it
    belong to the text it is cut from). Returns None when the box holds no
    text-like strokes.
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
        # A stroke belongs to the region that holds most of it, the mask without the couple of pixels
        # it was grown by: the foot of a handwritten gloss that dips into the (loose) box of the line
        # under it stays with the gloss (ch43 p1, 「これ」 above 「でぃめーとりあ」), a stroke of the
        # other region that sticks out of its box, or lies in the grown rim, is left out whole.
        taken = (exclude[cy1:cy2, cx1:cx2] > 0).astype(np.uint8)
        core = cv2.erode(taken, np.ones((5, 5), np.uint8))
        count, labels = cv2.connectedComponents(ink, connectivity=8)
        total = np.maximum(1, np.bincount(labels.ravel(), minlength=count))
        in_core = np.bincount(labels[core > 0], minlength=count) / total
        in_rim = np.bincount(labels[taken > 0], minlength=count) / total
        theirs = (in_core > 0.5) | (in_rim > 0.9)
        theirs[0] = False
        ink[theirs[labels]] = 0

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
    # Screentone: a box on a toned area takes in dozens of dots of one small size, more than the
    # text has strokes. They are tone, not text: not erased, not counted (ch42 p1, labels on a map).
    largest = max(c["size"] for c in seeds)
    tiny = max(3.0, 0.35 * largest)
    strokes = [c for c in seeds if c["size"] > tiny]
    dots = [c for c in seeds if c["size"] <= tiny]
    if strokes and len(dots) >= 15 and len(dots) > 1.5 * len(strokes):
        small = [c for c in comps if c["size"] <= tiny]
        tone = _tone(small, strokes, float(np.median([c["size"] for c in dots])), largest)
        comps = [c for c in comps if c["label"] not in tone]
        seeds = [c for c in seeds if c["label"] not in tone]
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
    vertical = direction.startswith("v")
    # Strokes inside the box that form a slanted line grow along it only: the labels beside a
    # label on a map are as close to it as its own next glyph.
    along = _slanted(labels, list(kept), char, vertical) if grow else None

    def off_line(c) -> bool:
        if along is None:
            return False
        point, normal, near = along[4]
        mx, my = (c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2
        return abs((mx - point[0]) * normal[0] + (my - point[1]) * normal[1]) > near

    # Along a slanted line, one step only: the strokes the box cut off at its ends. A dashed outline
    # the label is written along (a tent on a map) is spaced like glyphs and would be followed on.
    for _ in range((1 if along is not None else 12) if grow else 0):
        grew = False
        for c in comps:
            if (c["label"] in kept or c["touches"] or not c["textlike"] or c["size"] > 2 * char
                    or _gap(c["box"], union) > reach or off_line(c)):
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
    gap = max(2, int(round(0.15 * char)))

    def main_runs(runs: List[Tuple[int, int]], ink: np.ndarray) -> List[Tuple[int, int]]:
        # a line is as long as the others and holds some ink: a few strokes of an outline the box
        # took in (a dashed frame) are erased with the text but make no line
        widest = max((b - a for a, b in runs), default=0)
        heaviest = max((int(ink[a:b].sum()) for a, b in runs), default=0)
        return [(a, b) for a, b in runs if b - a >= 0.6 * widest and ink[a:b].sum() >= 0.15 * heaviest] or runs

    def split_lines(mask: np.ndarray):
        ink = mask.sum(axis=0 if vertical else 1)
        return ink, _split_touching(_runs(ink > 0, gap), mask, vertical, char)

    ink_per_row = sub.sum(axis=0 if vertical else 1)
    runs = _runs(ink_per_row > 0, gap)
    # one line: written on a slant? (a label along a road or a roof). If not, lines set so tight
    # that they touch are cut apart (a slanted line looks like such lines, row by row)
    slant = None
    if len(main_runs(runs, ink_per_row)) == 1:
        slant = _slanted(labels[uy1:uy2, ux1:ux2], list(kept), char, vertical)
    if slant is None:
        ink_per_row, runs = split_lines(sub)
        # Brackets drawn around a text of more lines are as tall as all of them (ch42 p1, a
        # handwritten "（資材 / 置き場？）") and join its lines into one; erased with the text, they make
        # no line. Only when the lines show without them: the stem of a big "ト" looks like one too.
        braces = [label for label, c in kept.items() if _bracket(c["box"], char, vertical)]
        if braces and len(braces) < len(kept):
            bare = np.isin(labels, [label for label in kept if label not in braces])[uy1:uy2, ux1:ux2]
            bare_ink, bare_runs = split_lines(bare.astype(np.uint8))
            shown = len(main_runs(bare_runs, bare_ink))
            if shown >= 2 and shown >= len(main_runs(runs, ink_per_row)):
                sub, ink_per_row, runs = bare, bare_ink, bare_runs
    main = main_runs(runs, ink_per_row)
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
    angle = 0.0
    if slant is not None:
        quad, across, angle, on_line, _axis = slant
        ox, oy = ux1 + cx1, uy1 + cy1
        lines = [[[round(float(px) + ox, 1), round(float(py) + oy, 1)] for px, py in quad]]
        font_size = int(round(across))
        if len(on_line) < len(kept):  # strokes of the labels next to it stay theirs
            kept = {label: kept[label] for label in on_line}
            keep_mask = np.isin(labels, list(kept)).astype(np.uint8)
            union = [min(c["box"][0] for c in kept.values()), min(c["box"][1] for c in kept.values()),
                     max(c["box"][2] for c in kept.values()), max(c["box"][3] for c in kept.values())]
            ux1, uy1, ux2, uy2 = union

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
                      strokes=len(kept), warning=warning, angle=round(angle, 2))
