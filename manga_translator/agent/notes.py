"""Translations set beside text kept as drawn (``note_region``).

Text written over detailed art cannot be erased without breaking the drawing. On ch42 p1, a map whose
labels sit on screentone between dashed lines, the inpaint review left 13 of its 30 areas for the human
(dashed lines cut, screentone smeared), and the labels typeset over them did not cover the damage. A
note keeps the original and sets the translation beside it in brackets, smaller: the way the user
glossed invented script by hand (「(我)」 beside the glyphs, ch36 and ch40).

The note is a region of its own (``Region.note_of``): nothing is erased under it, and the render draws
it with a white outline thicker than the page's (``NOTE_STROKE``), so it reads over screentone.
``place_note`` tries spots beside the lines (left of a vertical column, under a horizontal line, then
the other side), before and after them, and a step or two further out, and takes the one that covers
the least art without touching other text. Art is what stays after a blur: flat screentone blurs away
and the outlined note reads on it, line art does not (on the ch42 map the note of 「今ここ」 went over
the drawn faces beside it, which have fewer dark pixels than the tone around them). A label written on
a slant gets a note along the same slant.

Its size is 0.8 of the source text's, kept between 0.6 and 0.8 of the page's usual text size (``usual_size``:
the dialogue and narration, else the upper quarter of all sizes): on the ch42 map the labels' own sizes
ran from 9 to 109 px (the detector's boxes of slanted labels, white outlined lettering), and notes at
12-16 px over screentone could not be read, at 60-80 px they covered the map. A box the model gives only
says where the note goes: in the first real run on that map the model gave a box the size of each label
with every call, which made notes of 12-17 px, and seven calls were refused as too small.

Only text over art gets a note: ``flat_ground`` refuses text whose ground the inpaint fills with one
color (``inpaint.background``: plain paper or a flat color, no art line running into the text). On the
ch42 map that is 「今ここ」 in its box and 「北」 and 「井戸」 in their circles, which the user would
erase; the labels on the roads between dashed lines and in the buildings are not flat. Text on plain paper
that an outline only touches is on flat ground too (``PAPER``): in the second real run the model noted
an aside written beside a thought bubble on ch42 p6, whose outline touches it, though it had translated
it in its place the run before. A note the model moves with a box stays beside its text (``NEAR``): in
that run it moved notes 90-118 px from their labels on the map, where they no longer read as theirs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .page import BREAK, RegionError, with_breaks

OPEN, CLOSE = "（", "）"
NOTE_SCALE = 0.8
"""The note's font size against the source text's, and the largest against the page's usual size."""
SMALLEST = 0.6
"""The smallest note against the page's usual text size."""
MIN_NOTE_PX = 12
NOTE_STROKE = 0.2
"""The note's white outline (the render's stroke width, in font sizes)."""
BOX = {"h": (1.05, 0.5, 1.12, 0.5), "v": (1.0, 0.5, 1.2, 0.75)}
"""The note's box, in font sizes: (a, b, c, d) for a × characters + b along its lines and c × lines + d
across them. The render shrinks text whose box is smaller than it measures the text with its outline
(``rendering.calc_box_from_font`` at ``NOTE_STROKE`` on the user's fonts: one vertical line needs 1.9
font sizes across, a horizontal one 1.6) and grows vertical text in a larger box."""
INK = 140
"""Pixels darker than this are ink: the share the tool reports."""
DARK, ART = 0.3, 2.0
"""Weights of a spot's darkness (mean) and of its art (spread of the blurred page) in its cost."""
CROWDED = 0.02
"""A spot whose box covers more than this share of other text (or the text itself) is not free."""
NEAR = 2.5
"""How far from its text a note placed with a box may be, in the note's font sizes. The program's own
spots reach 2.2 (a gap of 0.2 and two steps out); on the ch42 map the model moved notes 3.7-4.9 away."""
COVERS = 0.1
"""Share of a text's lines a note placed with a box may cover: more hides the text (review 2026-11-26
P2-3: a note moved onto its own label covered all of it). One right beside it touches an edge."""
PAPER = 0.85
"""Share of plain paper within half a font size of the text's strokes from which the text is on flat
ground although a line touches it: 0.98 round the aside beside a thought bubble on ch42 p6, at most
0.54 round the map labels that need a note."""
_OPENERS = "（(【［[〔︵"
_CLOSERS = "）)】］]〕︶"
_NO_LINE_START = _CLOSERS + "、。，．・…‥！？!?ー"


def bracketed(text: str) -> str:
    """``text`` in full-width brackets, once: brackets written around all of it are dropped first."""
    body = with_breaks(text).strip()
    if len(body) >= 2 and body[0] in _OPENERS and body[-1] in _CLOSERS:
        body = body[1:-1].strip()
    if not body:
        raise RegionError("translation is empty")
    return OPEN + body + CLOSE


def note_lines(text: str, per_line: int) -> List[str]:
    """The lines of a note: the model's [BR] breaks, else even lines of at most ``per_line``
    characters, none starting with closing punctuation."""
    if BREAK in text:
        return [part for part in text.split(BREAK) if part]
    chars = list(text)
    count = max(1, math.ceil(len(chars) / max(1, per_line)))
    size = max(1, math.ceil(len(chars) / count))
    lines, i = [], 0
    while i < len(chars):
        j = min(len(chars), i + size)
        while j < len(chars) and chars[j] in _NO_LINE_START:
            j += 1
        lines.append("".join(chars[i:j]))
        i = j
    return lines


def _cells(line: str) -> float:
    """Length of a line in font sizes (``checks._cells``: Latin letters and digits are half)."""
    return sum(0.5 if ch.isascii() else 1.0 for ch in line if not ch.isspace()) or 1.0


@dataclass
class Frame:
    """Where a text lies: its middle, the way it runs and the way its lines follow each other."""
    center: np.ndarray
    along: np.ndarray
    """Unit vector along the lines: down a vertical column, right along a horizontal line."""
    across: np.ndarray
    """Unit vector across them: right of a column, down from a line."""
    length: float
    thickness: float


def _axes(direction: str, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    rad = math.radians(angle or 0.0)
    cos, sin = math.cos(rad), math.sin(rad)
    right, down = np.array([cos, sin]), np.array([-sin, cos])  # the page's x and y turned by ``angle``
    return (down, right) if direction == "v" else (right, down)


def text_frame(polygons: Sequence[np.ndarray], direction: str, angle: float = 0.0) -> Frame:
    along, across = _axes(direction, angle)
    points = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons])
    pu, pw = points @ along, points @ across
    center = along * (pu.min() + pu.max()) / 2 + across * (pw.min() + pw.max()) / 2
    return Frame(center, along, across, float(pu.max() - pu.min()), float(pw.max() - pw.min()))


@dataclass
class NotePlace:
    lines: List[List[List[float]]]
    """One quad per line (column), page pixels, as the region's ``lines``."""
    font_size: int
    direction: str
    angle: float
    text: str
    """The note as it is typeset: in brackets, [BR] where its lines break."""
    side: str
    """Where it went: left / right / above / below the text, or in the given box."""
    ink: float
    """Share of the note's box over ink: the art it covers."""
    crowded: bool = False
    """No free spot: the note covers other text."""
    stroke: float = NOTE_STROKE
    outgrows: bool = False
    """Larger than the box it was given (``place_in_box``)."""

    @property
    def bbox(self) -> List[int]:
        points = np.asarray(self.lines, dtype=np.float64).reshape(-1, 2)
        return [int(math.floor(points[:, 0].min())), int(math.floor(points[:, 1].min())),
                int(math.ceil(points[:, 0].max())), int(math.ceil(points[:, 1].max()))]


def _line_quads(center: np.ndarray, direction: str, angle: float, length: float, thickness: float,
                count: int) -> List[List[List[float]]]:
    """The note's box cut into ``count`` lines across: top to bottom, or columns right to left. Each
    quad goes round from the top left of its upright form, like the detector's."""
    along, across = _axes(direction, angle)
    right, down = (across, along) if direction == "v" else (along, across)
    step = thickness / count
    quads = []
    for k in range(count):
        offset = (thickness / 2 - (k + 0.5) * step) if direction == "v" else ((k + 0.5) * step - thickness / 2)
        middle = center + across * offset
        half_x, half_y = ((step / 2, length / 2) if direction == "v" else (length / 2, step / 2))
        corners = [middle - right * half_x - down * half_y, middle + right * half_x - down * half_y,
                   middle + right * half_x + down * half_y, middle - right * half_x + down * half_y]
        quads.append([[round(float(x), 1), round(float(y), 1)] for x, y in corners])
    return quads


def _box(center: np.ndarray, along: np.ndarray, across: np.ndarray, length: float, thickness: float) -> np.ndarray:
    a, w = along * length / 2, across * thickness / 2
    return np.array([center - a - w, center + a - w, center + a + w, center - a + w])


def _under(quad: np.ndarray, gray: np.ndarray, blocked: np.ndarray,
           smooth: Optional[np.ndarray] = None) -> Optional[Tuple[float, float, float]]:
    """(ink share, cost of the art, share over other text) under ``quad``; None when it leaves the
    page. ``smooth`` is ``gray`` blurred (``_smooth``)."""
    height, width = gray.shape[:2]
    if quad[:, 0].min() < 0 or quad[:, 1].min() < 0 or quad[:, 0].max() > width or quad[:, 1].max() > height:
        return None
    x1, y1 = int(math.floor(quad[:, 0].min())), int(math.floor(quad[:, 1].min()))
    x2, y2 = int(math.ceil(quad[:, 0].max())), int(math.ceil(quad[:, 1].max()))
    local = np.zeros((max(1, y2 - y1), max(1, x2 - x1)), np.uint8)
    cv2.fillPoly(local, [np.round(quad - (x1, y1)).astype(np.int32).reshape(-1, 1, 2)], 1)
    sel = local > 0
    sel = sel[:gray.shape[0] - y1, :gray.shape[1] - x1]
    if sel.sum() < 4:
        return None
    window = (slice(y1, y1 + sel.shape[0]), slice(x1, x1 + sel.shape[1]))
    under = gray[window][sel]
    covered = blocked[window][sel]
    art = float(smooth[window][sel].std()) / 255.0 if smooth is not None else float(under.std()) / 255.0
    cost = DARK * (1.0 - float(under.mean()) / 255.0) + ART * art
    return float((under < INK).mean()), cost, float((covered > 0).mean())


def _smooth(gray: np.ndarray, font: int) -> np.ndarray:
    """``gray`` blurred enough that screentone dots run together."""
    return cv2.GaussianBlur(gray, (0, 0), max(2.0, font / 10.0))


def flat_ground(image: np.ndarray, polygons: Sequence[np.ndarray], pad: int = 40,
                font_size: Optional[float] = None) -> bool:
    """Whether the ground under and around the text in ``polygons`` (its lines, page pixels) is flat in
    the inpaint's sense, so erasing it leaves the drawing whole, or plain paper that a line only touches
    (``PAPER``; ``font_size`` sets how far round the strokes, else the lines' thickness). Judged on the
    text's own strokes (the ink inside its lines, 3 px wider), not on what erasing takes: a fit's erase
    area reaches the circle round 「北」 on the ch42 map, and that line running into it made the ground
    look busy."""
    from .inpaint import FLAT_SPECKS, _kernel, background, ink

    height, width = image.shape[:2]
    points = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons])
    x1, y1 = (max(0, int(math.floor(v)) - pad) for v in points.min(axis=0))
    x2, y2 = int(math.ceil(points[:, 0].max())) + pad + 1, int(math.ceil(points[:, 1].max())) + pad + 1
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return False
    crop = np.ascontiguousarray(image[y1:y2, x1:x2])
    if crop.ndim == 2:
        crop = np.repeat(crop[:, :, None], 3, axis=2)
    crop = crop[:, :, :3]
    lines = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(lines, [np.round(np.asarray(p) - (x1, y1)).astype(np.int32).reshape(-1, 1, 2) for p in polygons], 1)
    inside = lines > 0
    strokes = ink(crop, inside) & inside
    if not strokes.any():
        return False
    near = cv2.dilate(strokes.astype(np.uint8), _kernel(3)) > 0
    ground = background(crop, near)
    if ground.flat:
        return True
    if ground.specks > FLAT_SPECKS:  # screentone
        return False
    if not font_size:
        font_size = float(np.median([min(cv2.minAreaRect(np.asarray(p, np.float32).reshape(-1, 2))[1])
                                     for p in polygons]))
    reach = max(6, int(round(0.5 * font_size)))
    ring = (cv2.dilate(strokes.astype(np.uint8), _kernel(reach)) > 0) & ~near
    if not ring.any():
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    paper = cv2.GaussianBlur(gray, (0, 0), 1.5) >= float(np.percentile(gray, 95)) - 30
    return float(paper[ring].mean()) >= PAPER


def gap(lines: Sequence[np.ndarray], polygons: Sequence[np.ndarray]) -> float:
    """Shortest distance in pixels between a note's lines and its text's (0 where they touch)."""
    note = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in lines]
    text = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons]
    points = np.concatenate(note + text)
    low = np.floor(points.min(axis=0)) - 2
    width, height = (np.ceil(points.max(axis=0)) - low + 3).astype(int)
    free = np.full((height, width), 255, np.uint8)
    cv2.fillPoly(free, [np.round(p - low).astype(np.int32).reshape(-1, 1, 2) for p in text], 0)
    distance = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    covered = np.zeros_like(free)
    cv2.fillPoly(covered, [np.round(p - low).astype(np.int32).reshape(-1, 1, 2) for p in note], 1)
    return float(distance[covered > 0].min()) if covered.any() else math.inf


def usual_size(reading: Sequence[float], every: Sequence[float]) -> Optional[float]:
    """The page's usual text size: the median of its dialogue and narration (``reading``) when there are
    two or more, else the upper quarter of all its sizes, which many small labels do not pull down (on the
    ch42 map the median of all was 21 px, the bubbles 30-35 px)."""
    reading = sorted(float(v) for v in reading if v)
    if len(reading) >= 2:
        return float(np.median(reading))
    every = sorted(float(v) for v in every if v)
    return float(np.percentile(every, 75)) if every else None


def note_font(source_px: float, page_px: Optional[float] = None) -> int:
    """The note's font size for source text of ``source_px``, on a page whose usual text size is
    ``page_px`` (see the module doc)."""
    font = NOTE_SCALE * float(source_px or 0)
    if page_px:
        font = min(max(font, SMALLEST * page_px), NOTE_SCALE * page_px)
    return max(MIN_NOTE_PX, int(round(font)))


def note_size(text: str, font_size: int, direction: str, length: float) -> Tuple[List[str], float, float]:
    """(lines, length along, thickness across) of the note at ``font_size``: lines no longer than the
    text beside it, or four characters."""
    per_line = max(4, math.ceil(length / max(1, font_size)))
    lines = note_lines(text, per_line)
    return (lines,) + _extent(lines, font_size, direction)


def _extent(lines: List[str], font_size: float, direction: str) -> Tuple[float, float]:
    """(along, across) of the note's box (``BOX``)."""
    a, b, c, d = BOX["v" if direction == "v" else "h"]
    return (a * max(_cells(line) for line in lines) + b) * font_size, (c * len(lines) + d) * font_size


def place_note(gray: np.ndarray, polygons: Sequence[np.ndarray], direction: str, angle: float, source_px: float,
               text: str, blocked: np.ndarray, page_px: Optional[float] = None) -> Optional[NotePlace]:
    """The note's place beside the text of ``polygons`` (see the module doc). ``blocked`` is a
    page-sized array of the pixels of other regions; the text's own lines are added here. None when no
    spot beside the text stays on the page."""
    direction = "v" if direction == "v" else "h"
    frame = text_frame(polygons, direction, angle)
    font = note_font(source_px, page_px)
    lines, length, thickness = note_size(text, font, direction, frame.length)
    smooth = _smooth(gray, font)
    gap = max(3.0, 0.2 * font)
    own = np.zeros(gray.shape[:2], np.uint8)
    cv2.fillPoly(own, [np.round(np.asarray(p)).astype(np.int32).reshape(-1, 1, 2) for p in polygons], 1)
    grow = max(1, int(gap * 0.4))  # the box keeps a quarter font size around the text as well
    own = cv2.dilate(own, np.ones((2 * grow + 1,) * 2, np.uint8))
    blocked = (blocked > 0) | (own > 0)

    # beside the lines first: left of a column (read after it), under a line; then the other side,
    # then after and before the text, then all of them a step further out
    first, second = (-1.0, 1.0) if direction == "v" else (1.0, -1.0)
    names = {("v", -1): "left", ("v", 1): "right", ("h", -1): "above", ("h", 1): "below"}
    beside = frame.thickness / 2 + gap + thickness / 2
    beyond = frame.length / 2 + gap + length / 2
    slack = (frame.length - length) / 2
    shifts = [0.0] + ([-slack, slack] if slack > font else [])
    end_names = {("v", 1): "below", ("v", -1): "above", ("h", 1): "right", ("h", -1): "left"}
    candidates = []
    for ring, extra in ((0, 0.0), (1, float(font)), (2, 2.0 * font)):
        for sign, preference in ((first, 0.0), (second, 0.03)):
            for shift in shifts:
                candidates.append((shift, sign * (beside + extra), preference + 0.06 * ring + (0.01 if shift else 0),
                                   names[(direction, int(sign))]))
        for sign, preference in ((1.0, 0.05), (-1.0, 0.07)):
            candidates.append((sign * (beyond + extra), 0.0, preference + 0.06 * ring, end_names[(direction, int(sign))]))

    best, fallback = None, None
    for du, dw, preference, side in candidates:
        center = frame.center + frame.along * du + frame.across * dw
        measured = _under(_box(center, frame.along, frame.across, length, thickness), gray, blocked, smooth)
        if measured is None:
            continue
        ink, art, covered = measured
        if covered <= CROWDED:
            cost = art + preference
            if best is None or cost < best[0]:
                best = (cost, center, side, ink)
        else:
            cost = 3 * covered + art + preference
            if fallback is None or cost < fallback[0]:
                fallback = (cost, center, side, ink)
    chosen, crowded = (best, False) if best is not None else (fallback, True)
    if chosen is None:
        return None
    _cost, center, side, ink = chosen
    quads = _line_quads(center, direction, angle, length, thickness, len(lines))
    return NotePlace(quads, font, direction, float(angle or 0.0), BREAK.join(lines), side, round(ink, 3), crowded)


def place_in_box(gray: np.ndarray, box: Sequence[float], direction: str, source_px: float, text: str,
                 page_px: Optional[float] = None) -> Optional[NotePlace]:
    """The note centered on a box the model gave (page pixels, upright) at its own size (``note_font``):
    the box says where, not how big. Of one to four lines, the fewest that fit the box, else the layout
    that outgrows it least; kept on the page. None when no layout fits on the page (review 2026-11-26
    P2-3: a long note was placed from x -288)."""
    direction = "v" if direction == "v" else "h"
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    length_room, thickness_room = (y2 - y1, x2 - x1) if direction == "v" else (x2 - x1, y2 - y1)
    font = note_font(source_px, page_px)
    layouts = [note_lines(text, 1)] if BREAK in text else \
        [note_lines(text, math.ceil(len(text) / count)) for count in range(1, 5)]
    height, width = gray.shape[:2]
    page_along, page_across = (height, width) if direction == "v" else (width, height)
    best = None
    for lines in layouts:
        length, thickness = _extent(lines, font, direction)
        if length > page_along or thickness > page_across:  # off the page whatever the box
            continue
        over = max(length / max(1.0, length_room), thickness / max(1.0, thickness_room))
        if over <= 1.0:
            best = (over, lines, length, thickness)
            break
        if best is None or over < best[0]:
            best = (over, lines, length, thickness)
    if best is None:
        return None
    over, lines, length, thickness = best
    half_x, half_y = (thickness / 2, length / 2) if direction == "v" else (length / 2, thickness / 2)
    center = np.array([min(max((x1 + x2) / 2, half_x), width - half_x),
                       min(max((y1 + y2) / 2, half_y), height - half_y)])
    along, across = _axes(direction, 0.0)
    measured = _under(_box(center, along, across, length, thickness), gray, np.zeros(gray.shape[:2], np.uint8))
    return NotePlace(_line_quads(center, direction, 0.0, length, thickness, len(lines)), font, direction, 0.0,
                     BREAK.join(lines), "box", round(measured[0], 3) if measured else 0.0, outgrows=over > 1.0)


WAYS = (("left", -1, 0), ("right", 1, 0), ("up", 0, -1), ("down", 0, 1))


def free_boxes(gray: np.ndarray, box: Sequence[float], direction: str, source_px: float, text: str,
               page_px: Optional[float], texts: Dict[int, Sequence[np.ndarray]], own: Sequence[np.ndarray],
               near: float) -> List[Tuple[str, int, List[int]]]:
    """The spots nearest a box the model gave (page pixels) whose note covers text: for each way (left,
    right, up, down), the shortest move of ``box`` after which its note (``place_in_box``) covers at
    most ``COVERS`` of every text in ``texts`` (id: its lines) and stays within ``near`` px of its own
    text (``own``). (way, pixels moved, the moved box), nearest first. The real run of 2026-11-27 (ch42
    p1): a box was refused for covering the note's own label, and the model left the note over the
    drawing it had wanted to clear, not knowing the spot past the label was free."""
    x1, y1, x2, y2 = (float(v) for v in box)
    step = max(2, int(round(note_font(source_px, page_px) / 4)))
    points = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in own])
    reach = 2 * near + max(x2 - x1, y2 - y1, *(points.max(axis=0) - points.min(axis=0)))
    found = []
    for way, ux, uy in WAYS:
        moved, last = step, None
        while moved <= reach:
            moved_box = [x1 + ux * moved, y1 + uy * moved, x2 + ux * moved, y2 + uy * moved]
            place = place_in_box(gray, moved_box, direction, source_px, text, page_px)
            if place is None or place.bbox == last:  # held at the page's edge
                break
            last = place.bbox
            if gap(place.lines, own) <= near and covered_text(place.lines, texts)[1] <= COVERS:
                found.append((way, moved, [int(round(v)) for v in moved_box]))
                break
            moved += step
    return sorted(found, key=lambda spot: spot[1])


def covered_text(lines: Sequence[np.ndarray], texts: Dict[int, Sequence[np.ndarray]]) -> Tuple[Optional[int], float]:
    """The text a note's ``lines`` cover most, of ``texts`` (id: its lines, page pixels): its id and the
    share of its lines under the note; (None, 0.0) when it covers none. A note right beside its text
    touches it at most along an edge."""
    note = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in lines]
    if not note:
        return None, 0.0
    points = np.concatenate(note)
    nx1, ny1 = points.min(axis=0)
    nx2, ny2 = points.max(axis=0)
    best: Tuple[Optional[int], float] = (None, 0.0)
    for rid, polygons in texts.items():
        text = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in polygons]
        if not text:
            continue
        both = np.concatenate([points] + text)
        tx1, ty1 = np.concatenate(text).min(axis=0)
        tx2, ty2 = np.concatenate(text).max(axis=0)
        if tx2 <= nx1 or nx2 <= tx1 or ty2 <= ny1 or ny2 <= ty1:
            continue
        low = np.floor(both.min(axis=0))
        width, height = (np.ceil(both.max(axis=0)) - low + 2).astype(int)
        drawn = np.zeros((height, width), np.uint8)
        cv2.fillPoly(drawn, [np.round(p - low).astype(np.int32).reshape(-1, 1, 2) for p in text], 1)
        under = np.zeros_like(drawn)
        cv2.fillPoly(under, [np.round(p - low).astype(np.int32).reshape(-1, 1, 2) for p in note], 1)
        total = int(drawn.sum())
        share = float((drawn & under).sum()) / total if total else 0.0
        if share > best[1]:
            best = (rid, share)
    return best
