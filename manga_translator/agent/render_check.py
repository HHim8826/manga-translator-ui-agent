"""The text as the render draws it, checked when the agent finishes a page (``mark_page_done``).

The program checks before it only estimate the layout from the box and the font size. This one renders
every region the agent typeset alone, with the user's render settings and the page's bubble mask (the
font check's renderer, ``fonts.PipelineFontBackend``), and sets the size of the drawn glyphs against

- the size of the source strokes the translation replaces (``snap_text`` on the original), and
- the same source text drawn elsewhere on the page,

and the angle it is drawn at against the slant of the source lines (a slanted label typeset upright).

Both sizes are measured the same way, so what the measuring gets wrong on one side it mostly gets wrong
on the other. Two lines that touch are read as one line twice as thick; the size is then the extent
across the lines divided by the lines there are: the translation's line breaks, and the bands across
that the source's detected lines cover (the detector cuts a column in two along it as often as it finds
two columns). A dotted ellipsis drawn in a column is measured as thin as its dots.

ch42 p1 (the README's hard page): a label cut in two by the detector drawn at 54 px over 25 px strokes,
a note squeezed to 10 px over 21 px strokes, and a slanted 「テント屋根」 typeset upright at 40 px while
its five copies on the map are drawn at 19 px. On chapters 38-43 the measuring alone, without the line
count, mistook 19 of 27 regions; with it, 3 of 14 (source lines the detector took as one, screentone
under the source). The angle: 7 of 602, all slanted labels or sound effects drawn upright."""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from .checks import WEIGHTS, Issue
from .snap import snap_text

if TYPE_CHECKING:  # pragma: no cover
    from .page import PageDoc, Region

LARGE = 1.6
SMALL = 0.6
"""Drawn this many times the size of the source text (or of the same text elsewhere) or more / less."""
MIN_SOURCE_PX = 10
"""Smaller source strokes are not measured well enough (dots, a bit of a stroke on screentone)."""
LINE_PITCH = 1.1
"""A line takes this many font sizes across the lines, the gap with it."""
SAME_SOURCE = 1.35
"""The same text elsewhere counts only when its source strokes are about as big."""
TURNED = 15
"""Source lines slanted this much or more, drawn this many degrees off. Not the other way round: the
measuring misses a slant more often than it finds one that is not there."""
_PLAIN = re.compile(r"[\s？?！!…・、。]")
_DOTS = re.compile(r"[…‥⋯︙︰]")
COVER = 0.7
"""A snap across less than this share of the source lines' extent found one of the lines, not all of
them run together."""


@dataclass
class Drawn:
    drawn: float
    """Size of the drawn glyphs, px."""
    source: Optional[float]
    """Size of the source strokes, px; None when they could not be measured."""
    source_angle: Optional[float] = None
    """Slant of the source lines, degrees; None when they could not be measured."""
    angle: float = 0.0
    """The angle the region is drawn at (its JSON ``angle``)."""


def measured_size(snap, lines: int, direction: str, extent: Optional[float] = None) -> Optional[float]:
    """Font size from a ``SnapResult`` of text that has ``lines`` lines (columns when ``direction`` is
    ``v``) across ``extent`` px: a snap that found fewer lines took touching lines for one, so the size
    is the extent across them divided by the lines."""
    if snap is None or snap.font_size < 6:
        return None
    x1, y1, x2, y2 = snap.bbox
    across = (x2 - x1) if direction == "v" else (y2 - y1)
    if abs(snap.angle) >= 5 or len(snap.lines) >= lines or (extent and across < COVER * extent):
        return float(snap.font_size)
    return min(float(snap.font_size), across / (max(1, lines) * LINE_PITCH))


def lines_across(polygons, direction: str) -> tuple:
    """(lines, extent px) across the detected lines of a region (columns when ``direction`` is ``v``):
    pieces that cover the same band across count once."""
    axis = 0 if direction == "v" else 1
    points = [np.asarray(q, dtype=np.float64).reshape(-1, 2) for q in polygons]
    bands = sorted((float(p[:, axis].min()), float(p[:, axis].max())) for p in points if len(p))
    if not bands:
        return 1, 0.0
    count, (low, high) = 1, bands[0]
    for a, b in bands[1:]:
        if min(high, b) - max(low, a) >= 0.5 * min(high - low, b - a):
            low, high = min(low, a), max(high, b)
        else:
            count, (low, high) = count + 1, (a, b)
    return count, bands[-1][1] - bands[0][0]


def _angle(region: "Region") -> float:
    try:
        return float(region.data.get("angle") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _lines(text: str) -> int:
    return text.count("[BR]") + text.count("\n") + 1


def _box(region: "Region") -> tuple:
    """The detected lines' own box, not ``Region.bbox`` rounded outwards: a pixel more takes in a
    stroke beside it, and a slanted label is measured as two upright lines (ch42 p1 #14)."""
    points = np.concatenate(region.polygons) if region.polygons else np.zeros((1, 2))
    return (float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()),
            float(points[:, 1].max()))


class RenderCheck:
    """Measures the regions of a page as the render draws them (see the module doc)."""

    def __init__(self, backend):
        self.backend = backend
        self._source: Dict[tuple, tuple] = {}
        """(image, box, direction, lines, extent) -> source (size, slant): the strokes do not change
        during a run."""

    def source(self, image: np.ndarray, image_path: str, region: "Region") -> tuple:
        """(size px, slant degrees) of the source strokes, each None when not measured."""
        direction = "v" if region.direction.startswith("v") else "h"
        lines, extent = lines_across(region.polygons, direction)
        box = _box(region)
        key = (image_path, box, direction, lines, round(extent))
        if key not in self._source:
            snap = snap_text(image, box, direction, grow=False)
            size = measured_size(snap, lines, direction, extent)
            self._source[key] = (size, float(snap.angle) if size is not None else None)
        return self._source[key]

    async def measure(self, doc: "PageDoc") -> Dict[int, Drawn]:
        """Region id -> sizes, for the translated regions the agent typeset (not the user's locked ones,
        not the ones a check left as they were)."""
        ids = [r.id for r in doc.active_regions() if not r.skipped]
        page = doc.json_data()[doc.key]
        loaded = await self.backend.load_page(doc.image_path, page)
        if loaded is None:
            return {}
        blocks, image = loaded
        if len(blocks) != len(ids):  # a region failed to load: numbers would not match
            return {}
        out: Dict[int, Drawn] = {}
        for rid, block in zip(ids, blocks):
            region = doc.regions[rid]
            if not region.translation or region.locked or region.unchanged:
                continue
            drawn_image = await self.backend.render_alone(doc.image_path, block)
            if drawn_image is None:
                continue
            direction = "v" if region.direction.startswith("v") else "h"
            x1, y1, x2, y2 = region.bbox
            grow_x, grow_y = (x2 - x1) * 0.5 + 10, (y2 - y1) * 0.5 + 10
            snap = snap_text(drawn_image, (x1 - grow_x, y1 - grow_y, x2 + grow_x, y2 + grow_y), direction, grow=False)
            drawn = measured_size(snap, _lines(region.translation), direction)
            if drawn is None:
                continue
            size, slant = self.source(image, doc.image_path, region)
            out[rid] = Drawn(drawn=drawn, source=size, source_angle=slant, angle=_angle(region))
        return out

    async def issues(self, doc: "PageDoc") -> List[Issue]:
        return size_issues(doc, await self.measure(doc))


def _dotted(translation: str) -> bool:
    """An ellipsis with one glyph or none besides: measured as thin as its dots."""
    text = (translation or "").replace("[BR]", "")
    return bool(_DOTS.search(text)) and len(_PLAIN.sub("", _DOTS.sub("", text))) <= 1


def size_issues(doc: "PageDoc", sizes: Dict[int, Drawn]) -> List[Issue]:
    """``drawn_large`` / ``drawn_small`` against the source strokes, then ``drawn_unlike`` against the
    same text elsewhere on the page (the regions not already listed), then ``drawn_turned`` (the
    regions with nothing else listed)."""
    issues: List[Issue] = []
    listed = set()
    for rid, size in sizes.items():
        if size.source is None or size.source < MIN_SOURCE_PX:
            continue
        ratio = size.drawn / size.source
        detail = {"drawn": int(round(size.drawn)), "source": int(round(size.source))}
        if ratio >= LARGE:
            issues.append(Issue("drawn_large", rid, {**detail, "times": round(ratio, 1)}, WEIGHTS["drawn_large"]))
            listed.add(rid)
        elif ratio <= SMALL and not _dotted(doc.regions[rid].translation):
            issues.append(Issue("drawn_small", rid, {**detail, "percent": int(round(100 * ratio))},
                                WEIGHTS["drawn_small"]))
            listed.add(rid)
    groups: Dict[str, List[int]] = {}
    for rid, size in sizes.items():
        key = _PLAIN.sub("", doc.regions[rid].source_text or doc.regions[rid].ocr_local or "")
        if len(key) >= 2 and size.source:
            groups.setdefault(key, []).append(rid)
    for key, members in groups.items():
        for rid in members:
            if rid in listed:
                continue
            others = [m for m in members if m != rid and m not in listed]
            if len(others) < 2:
                continue
            drawn = statistics.median(sizes[m].drawn for m in others)
            source = statistics.median(sizes[m].source for m in others)
            if not 1 / SAME_SOURCE <= sizes[rid].source / source <= SAME_SOURCE:
                continue  # drawn that much bigger or smaller in the source too
            if not (sizes[rid].drawn >= LARGE * drawn or sizes[rid].drawn <= SMALL * drawn):
                continue
            issues.append(Issue("drawn_unlike", rid, {"drawn": int(round(sizes[rid].drawn)),
                                                      "other": int(round(drawn)), "text": key[:12]},
                                WEIGHTS["drawn_unlike"]))
    listed = {i.region for i in issues}
    for rid, size in sizes.items():
        if rid in listed or size.source_angle is None or size.source is None or size.source < MIN_SOURCE_PX:
            continue
        off = abs(size.source_angle - size.angle) % 180
        if abs(size.source_angle) >= TURNED and min(off, 180 - off) >= TURNED:
            issues.append(Issue("drawn_turned", rid, {"source": int(round(size.source_angle)),
                                                      "drawn": int(round(size.angle))}, WEIGHTS["drawn_turned"]))
    return sorted(issues, key=lambda i: i.region or 0)
