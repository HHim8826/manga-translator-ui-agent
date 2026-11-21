"""One page's ``*_translations.json`` as the agent edits it.

Regions get stable ids (1, 2, ...) in the order the preprocess wrote them; the
ids stay valid for the whole run even after merges. Agent-only data (readings,
reasons, flags) goes to the sidecar, never into the region fields that the
existing loaders read.

On save:

* skipped regions (text that is kept as it is, such as a background sign) are
  left out of the JSON and their text is taken out of the saved mask, so the
  render step neither erases nor covers them;
* merged regions are written as one region whose lines are the union;
* regions added by the agent or the recall check (text the detector missed)
  are written like detector regions, and their strokes are added to the mask
  so the render step erases them.
"""
from __future__ import annotations

import base64
import copy
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from .snap import SnapResult
from .store import write_json_atomic

CATEGORIES = ("dialogue", "narration", "sfx", "sign", "handwritten", "note", "other")
FLAG_TYPES = ("uncertain_ocr", "uncertain_translation", "layout", "inpaint", "missing_text",
              "low_confidence", "step_limit", "error", "other")
BREAK = "[BR]"
"""The app's line break marker in translations (the editor writes it too); the renderer breaks there."""
_BREAK_MARKS = re.compile(r"\s*(?:\[BR\]|【BR】|<br\s*/?>|\r?\n)\s*", re.IGNORECASE)
MIN_BREAK_CHARS = 4
"""Translations shorter than this (letters and digits) need no line break even for a multi-line source."""


def with_breaks(text: str) -> str:
    """``text`` with every line break written as ``[BR]``: newlines and the other spellings become
    ``[BR]``, doubled markers one, and none is left at the start or the end."""
    parts = _BREAK_MARKS.split(str(text or ""))
    return BREAK.join(part for part in parts if part.strip())


def break_count(text: str) -> int:
    return str(text or "").count(BREAK)


def text_length(text: str) -> int:
    """Letters and digits only: punctuation squeezes in (「不行。」 is 2)."""
    return sum(1 for ch in str(text or "").replace(BREAK, "") if ch.isalnum())


def needs_break(line_count: int, translation: str) -> bool:
    """A multi-line source whose translation would be typeset as one line (AI line breaks)."""
    return line_count >= 2 and text_length(translation) >= MIN_BREAK_CHARS and not break_count(translation)


PARAGRAPH_GAP = 0.5
"""A blank space between two regions at least this many line widths wide separates paragraphs.
Measured on chapter 34: lines of one paragraph are 2-9 px apart at 30-40 px wide; paragraphs side by
side are 32-51 px apart, and the two blocks of a stepped narration box 45-65 px."""


def paragraph_break(regions: List["Region"]) -> Optional[tuple]:
    """Two of the regions that a blank space between paragraphs separates, or None when the regions
    form one block. Regions join when their outlines are closer than a paragraph gap both across the
    lines (side by side columns of vertical text, rows of horizontal text) and along them (a block
    below another one, as in a stepped narration box); the regions must all join up."""
    vertical = sum(1 for r in regions if r.direction.startswith("v")) * 2 >= len(regions)
    across = (0, 2) if vertical else (1, 3)  # the axis the lines follow each other on
    along = (1, 3) if vertical else (0, 2)
    blocks, widths, inner = [], [], []
    for region in regions:
        boxes = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in region.polygons]
        if not boxes:
            continue
        widths += [b[across[1]] - b[across[0]] for b in boxes]
        boxes.sort(key=lambda b: b[across[0]])
        inner += [b[across[0]] - a[across[1]] for a, b in zip(boxes, boxes[1:]) if b[across[0]] > a[across[1]]]
        blocks.append((region, tuple(f(b[k] for b in boxes) for k, f in ((0, min), (1, min), (2, max), (3, max)))))
    if len(blocks) < 2:
        return None
    width = float(np.median(widths))
    limit_across = max(PARAGRAPH_GAP * width, 3 * float(np.median(inner)) if inner else 0.0)
    limit_along = PARAGRAPH_GAP * width

    def gap(a, b, axis) -> float:
        return max(a[axis[0]], b[axis[0]]) - min(a[axis[1]], b[axis[1]])

    def apart(a, b) -> bool:
        return gap(a, b, across) >= limit_across or gap(a, b, along) >= limit_along

    joined = {0}
    grown = True
    while grown:
        grown = False
        for j in range(len(blocks)):
            if j not in joined and any(not apart(blocks[i][1], blocks[j][1]) for i in joined):
                joined.add(j)
                grown = True
    if len(joined) == len(blocks):
        return None
    i, j = min(((i, j) for i in joined for j in range(len(blocks)) if j not in joined),
               key=lambda p: max(gap(blocks[p[0]][1], blocks[p[1]][1], across),
                                 gap(blocks[p[0]][1], blocks[p[1]][1], along)))
    return blocks[i][0], blocks[j][0]


def line_groups(region: "Region") -> List[List[int]]:
    """The region's lines (indexes into its polygons) in groups that a blank space between paragraphs
    separates, in line order: two bubbles, or the two parts of one joined bubble, that the detector
    boxed as one text. One group when the lines form one block. Ch36 p7: the two lines of a joined
    bubble stood 46 px apart at 32 px wide, where lines of one paragraph are 2-9 px apart."""
    polys = region.polygons
    if len(polys) < 2:
        return [list(range(len(polys)))]
    vertical = region.direction.startswith("v")
    across = (0, 2) if vertical else (1, 3)
    along = (1, 3) if vertical else (0, 2)
    boxes = [(p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()) for p in polys]
    limit = PARAGRAPH_GAP * float(np.median([b[across[1]] - b[across[0]] for b in boxes]))

    def gap(a, b, axis) -> float:
        return max(a[axis[0]], b[axis[0]]) - min(a[axis[1]], b[axis[1]])

    group = list(range(len(boxes)))

    def root(i: int) -> int:
        while group[i] != i:
            i = group[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if gap(boxes[i], boxes[j], across) < limit and gap(boxes[i], boxes[j], along) < limit:
                group[root(j)] = root(i)
    groups: Dict[int, List[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(root(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


@dataclass
class Region:
    id: int
    data: dict
    ocr_local: str = ""
    source_text: Optional[str] = None
    translation: Optional[str] = None
    ruby: str = ""
    category: str = ""
    confidence: Optional[float] = None
    skipped: bool = False
    skip_category: str = ""
    merged_into: Optional[int] = None
    link_group: Optional[int] = None
    reasons: List[str] = field(default_factory=list)
    origin: str = "detector"
    """``detector``, ``agent`` (add_region) or ``recall`` (the recall check)."""
    origin_kind: str = ""
    """What the recall check thought the text was (dialogue, sfx, ...)."""
    erase: Optional[tuple] = None
    """Added regions: (x, y, 0/1 patch) of pixels to add to the mask."""
    font: str = ""
    """The font of the region's style; empty keeps the JSON's font (the user's)."""
    style: str = ""
    """The agent's font style for this region (``font_styles.py``); empty for normal dialogue."""
    # --- a page checked again (``revise.py``) ---------------------------------
    user_edited: bool = False
    """The user changed the region in the editor after the agent (its text or place), or drew it."""
    user_font: bool = False
    """The user gave the region its font in the editor: the thin-font replacement and the font check
    leave it."""
    locked: bool = False
    """The user's region, kept as it is: the write tools refuse it (setting ``revise_lock_user_edits``)."""
    loaded: Optional[dict] = None
    """The region's JSON as the check loaded it; written back unchanged while the agent leaves the
    region's text alone, so the editor's own fields stay."""
    loaded_state: Optional[tuple] = None
    """(source text, translation, lines) at load."""
    was_skipped: bool = False
    """Skipped by the earlier run: its text is not in the saved mask."""
    drop_rich: bool = False
    """The region's ``translation_rich`` only drew glyphs from a fallback of its old font; the font
    check makes it again for the new font."""

    @property
    def polygons(self) -> List[np.ndarray]:
        return [np.asarray(quad, dtype=np.float64).reshape(-1, 2) for quad in self.data.get("lines", [])]

    @property
    def bbox(self) -> List[int]:
        pts = np.concatenate(self.polygons) if self.polygons else np.zeros((1, 2))
        return [int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min())),
                int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max()))]

    @property
    def center(self) -> List[float]:
        x1, y1, x2, y2 = self.bbox
        return [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)]

    @property
    def line_count(self) -> int:
        """Lines (columns for vertical text) of the source text; merged regions have the union."""
        return len(self.data.get("lines") or [])

    @property
    def direction(self) -> str:
        return str(self.data.get("direction") or "auto")

    @property
    def font_size(self) -> int:
        try:
            return int(self.data.get("font_size") or 0)
        except (TypeError, ValueError):
            return 0

    @property
    def active(self) -> bool:
        return self.merged_into is None

    @property
    def done(self) -> bool:
        return self.skipped or bool(self.translation) or self.locked

    @property
    def unchanged(self) -> bool:
        """A loaded region the agent left as it was."""
        return self.loaded_state is not None and not self.skipped and \
            self.loaded_state == (self.source_text, self.translation, self.data.get("lines"))

    def status(self) -> str:
        if self.merged_into is not None:
            return f"merged into #{self.merged_into}"
        if self.skipped:
            return f"skipped ({self.skip_category})"
        return "translated" if self.translation else "pending"


STYLE_KEYS = ("target_lang", "source_lang", "line_spacing", "letter_spacing", "stroke_width", "font_family")
"""Copied from an existing region of the page into an added region."""


class RegionError(ValueError):
    """Bad tool arguments; the message is sent back to the model."""


class PageDoc:
    def __init__(self, image_path: str, json_path: str, raw: dict):
        if not raw:
            raise ValueError(f"empty translation file: {json_path}")
        self.image_path = image_path
        self.json_path = json_path
        self.key = next(iter(raw))
        page_data = raw[self.key]
        if isinstance(page_data, list):  # old format: a bare region list
            page_data = {"regions": page_data}
        self.page_data: dict = page_data
        self.regions: Dict[int, Region] = {}
        for index, data in enumerate(page_data.get("regions", []), start=1):
            local = data.get("text") or "".join(data.get("texts") or [])
            self.regions[index] = Region(id=index, data=copy.deepcopy(data), ocr_local=str(local or ""))
        self.order: List[int] = list(self.regions)
        self.flags: List[dict] = []
        self.status = "pending"
        self.confidence: Optional[float] = None
        """Page confidence: the lower of the agent's number and the program check score."""
        self.agent_confidence: Optional[float] = None
        self.check_score: Optional[float] = None
        self.checks: List[dict] = []
        """Program check issues left when the page was finished (see checks.py)."""
        self.note = ""
        self._history: List[tuple] = []
        self._next_link = 1
        self.revise = False
        """Loaded by ``revise.load_page``: a page an earlier run finished, checked again."""
        self.removed: List[dict] = []
        """``{"bbox", "text"}`` of boxes the user deleted in the editor (a page checked again): their
        text stays as the user left it, so neither the recall check nor add_region boxes it again."""
        self.sidecar_extra: Dict[str, object] = {}
        """Sidecar fields of the earlier run that the check keeps (the inpaint review's)."""

    @classmethod
    def load(cls, image_path: str, json_path: Optional[str] = None) -> "PageDoc":
        if json_path is None:
            from ..utils.path_manager import find_json_path
            json_path = find_json_path(image_path)
        if not json_path or not os.path.exists(json_path):
            raise FileNotFoundError(f"no translation JSON for {image_path}")
        import json
        with open(json_path, "r", encoding="utf-8") as fh:
            return cls(image_path, json_path, json.load(fh))

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.image_path))[0]

    # --- lookup -------------------------------------------------------------
    def get(self, rid) -> Region:
        try:
            rid = int(str(rid).lstrip("#"))
        except ValueError:
            raise RegionError(f"region id must be a number like 3, got {rid!r}")
        region = self.regions.get(rid)
        if region is None:
            raise RegionError(f"there is no region #{rid} on this page (ids: {', '.join(map(str, self.order))})")
        if region.merged_into is not None:
            raise RegionError(f"region #{rid} was merged into #{region.merged_into}; use #{region.merged_into}")
        return region

    def _final(self, region: Region) -> Region:
        """The active region that ``region`` ended up in after merges."""
        seen = set()
        while region.merged_into is not None and region.id not in seen:
            seen.add(region.id)
            region = self.regions.get(region.merged_into, region)
        return region

    def active_regions(self) -> List[Region]:
        return [self.regions[i] for i in self.order if self.regions[i].active]

    def missing(self) -> List[int]:
        return [r.id for r in self.active_regions() if not r.done]

    def boxes(self) -> List[list]:
        """``[id, x1, y1, x2, y2, state]`` of each active region, for the Agents page preview;
        state is ``pending``, ``done``, ``skipped`` or ``flagged``."""
        flagged = {f.get("region") for f in self.flags}
        out = []
        for r in self.active_regions():
            state = "skipped" if r.skipped else "flagged" if r.id in flagged else \
                "done" if r.translation else "pending"
            out.append([r.id, *r.bbox, state])
        return out

    # --- history ------------------------------------------------------------
    def _checkpoint(self, label: str) -> None:
        # Flags are notes for the human, not edits: undo leaves them (one on a region that undo
        # removes goes with it).
        self._history.append((label, copy.deepcopy(self.regions), list(self.order)))

    def clear_history(self) -> None:
        self._history.clear()

    def undo(self) -> str:
        if not self._history:
            raise RegionError("nothing to undo on this page")
        label, self.regions, self.order = self._history.pop()
        self.flags = [f for f in self.flags if f.get("region") is None or f.get("region") in self.regions]
        return label

    # --- edits --------------------------------------------------------------
    @staticmethod
    def _unlocked(region: Region) -> Region:
        if region.locked:
            raise RegionError(f"#{region.id} is the user's: they changed it in the editor after the earlier run, "
                              f"so it stays as it is. Leave it; if something in it looks wrong, flag_for_review "
                              f"#{region.id} with the reason.")
        return region

    @staticmethod
    def _agents(region: Region) -> None:
        """The agent changed the region (the lock is off): it is no longer the user's version."""
        region.user_edited = False

    def set_text(self, rid, source_text: str, translation: str, category: str = "dialogue",
                 confidence: Optional[float] = None, ruby: str = "", reason: str = "") -> Region:
        region = self._unlocked(self.get(rid))
        translation = with_breaks(translation)
        if not translation.strip():
            raise RegionError(f"#{region.id}: translation is empty; use skip_region for text that stays as it is")
        if category and category not in CATEGORIES:
            raise RegionError(f"category must be one of {', '.join(CATEGORIES)}")
        self._checkpoint(f"set_text #{region.id}")
        self._agents(region)
        region.source_text = (source_text or "").strip()
        region.translation = translation
        region.ruby = (ruby or "").strip()
        region.category = category or region.category or "dialogue"
        region.confidence = None if confidence is None else max(0.0, min(1.0, float(confidence)))
        region.skipped = False
        region.skip_category = ""
        if reason:
            region.reasons.append(reason)
        return region

    def skip(self, rid, category: str, reason: str) -> Region:
        region = self._unlocked(self.get(rid))
        if not reason:
            raise RegionError("reason is required")
        self._checkpoint(f"skip #{region.id}")
        self._agents(region)
        region.skipped = True
        region.skip_category = category or "other"
        region.translation = None
        region.reasons.append(reason)
        return region

    def merge(self, ids, reason: str) -> Region:
        regions = [self._unlocked(self.get(i)) for i in ids]
        unique = {r.id: r for r in regions}
        if len(unique) < 2:
            raise RegionError("merge_regions needs at least two different ids")
        if not reason:
            raise RegionError("reason is required")
        ordered = [self.regions[i] for i in self.order if i in unique]
        split = paragraph_break(ordered)
        if split is not None:
            raise RegionError(
                f"#{split[0].id} and #{split[1].id} are separate blocks of text (a blank space lies between them, "
                "beside or below), so they stay two regions even in one bubble or narration box: translate each "
                "where it is, and use link_regions if one sentence runs across them. Merge only boxes that cut "
                "one paragraph apart.")
        self._checkpoint("merge " + "+".join(f"#{r.id}" for r in ordered))
        for region in ordered:
            self._agents(region)
        keep = ordered[0]
        lines, texts, local = [], [], []
        for region in ordered:
            lines.extend(region.data.get("lines", []))
            texts.extend(region.data.get("texts") or [region.ocr_local])
            local.append(region.ocr_local)
        keep.data["lines"] = lines
        keep.data["texts"] = [t for t in texts if t]
        keep.data["font_size"] = max((r.font_size for r in ordered), default=keep.font_size)
        keep.data.pop("render_box_rect_local", None)
        keep.data["center"] = keep.center
        keep.ocr_local = "\n".join(t for t in local if t)
        keep.source_text = keep.translation = None
        keep.skipped = False
        keep.reasons.append(reason)
        for region in ordered[1:]:
            region.merged_into = keep.id
        return keep

    def split(self, rid, box, reason: str) -> Region:
        """Move the lines of ``rid`` whose middle lies inside ``box`` (page pixels) to a new region, for
        one detected box that holds two separate texts (a small aside beside the main text, a subtitle
        printed in a logo above a chapter title). Both parts are then translated or skipped on their own;
        the new region comes right after the old one in reading order."""
        region = self._unlocked(self.get(rid))
        if not reason:
            raise RegionError("reason is required")
        x1, y1, x2, y2 = box
        polys = region.polygons
        inside = [i for i, p in enumerate(polys)
                  if x1 <= p[:, 0].mean() <= x2 and y1 <= p[:, 1].mean() <= y2]
        if not inside or len(inside) == len(polys):
            listed = "; ".join(
                f"line {i + 1} [{int(p[:, 0].min())}, {int(p[:, 1].min())}, {int(p[:, 0].max())}, {int(p[:, 1].max())}]"
                for i, p in enumerate(polys))
            if len(polys) < 2:
                raise RegionError(f"#{region.id} is one detected line, and split_region moves whole lines: translate "
                                  "it as one region, and flag_for_review if its two texts need to be apart")
            raise RegionError(f"the box holds {'none' if not inside else 'all'} of the lines of #{region.id}; "
                              f"whole lines move and both parts keep at least one. #{region.id} has {len(polys)} "
                              f"lines: {listed}. Give a box around the middle of the lines for the new region.")
        self._checkpoint(f"split #{region.id}")
        self._agents(region)
        vertical = region.direction.startswith("v")

        def thickness(quads) -> float:
            sizes = [np.ptp(np.asarray(q, dtype=np.float64).reshape(-1, 2)[:, 0 if vertical else 1]) for q in quads]
            return float(np.median(sizes)) if sizes else 0.0

        lines = region.data.get("lines", [])
        moved = [lines[i] for i in inside]
        rest = [line for i, line in enumerate(lines) if i not in set(inside)]
        whole = thickness(lines) or 1.0
        data = copy.deepcopy(region.data)
        for part, quads in ((data, moved), (region.data, rest)):
            part["lines"] = quads
            if region.font_size:  # a small aside gets a smaller size than the text beside it
                part["font_size"] = max(6, int(round(region.font_size * min(1.0, thickness(quads) / whole))))
            part.pop("render_box_rect_local", None)
        data["texts"], data["text"] = [], ""
        new_id = max(self.regions) + 1
        new = Region(id=new_id, data=data, origin=region.origin, origin_kind=region.origin_kind,
                     erase=region.erase, reasons=[reason])
        data["center"] = new.center
        region.data["center"] = region.center
        region.source_text = region.translation = None
        region.skipped = False
        region.skip_category = ""
        region.reasons.append(reason)
        self.regions[new_id] = new
        self.order.insert(self.order.index(region.id) + 1, new_id)
        return new

    def set_order(self, ids) -> List[int]:
        wanted = [self.get(i).id for i in ids]
        if len(set(wanted)) != len(wanted):
            raise RegionError("set_reading_order: an id appears twice")
        self._checkpoint("set_reading_order")
        rest = [i for i in self.order if i not in wanted]
        self.order = wanted + rest
        return self.order

    def link(self, ids, reason: str) -> int:
        regions = [self.get(i) for i in ids]
        if len({r.id for r in regions}) < 2:
            raise RegionError("link_regions needs at least two different ids")
        self._checkpoint("link " + "+".join(f"#{r.id}" for r in regions))
        group = self._next_link
        self._next_link += 1
        for region in regions:
            region.link_group = group
            if reason:
                region.reasons.append(reason)
        return group

    def set_line_breaks(self, rid, text: str, reason: str = "") -> Region:
        region = self._unlocked(self.get(rid))
        if not region.translation:
            raise RegionError(f"#{region.id} has no translation yet; call set_text first")
        text = with_breaks(text)
        if "".join(text.replace(BREAK, "").split()) != "".join(region.translation.replace(BREAK, "").split()):
            raise RegionError(f"#{region.id}: the text must be the current translation with line breaks only")
        self._checkpoint(f"line_breaks #{region.id}")
        self._agents(region)
        region.translation = text
        if reason:
            region.reasons.append(reason)
        return region

    def flag(self, rid, type: str, reason: str) -> dict:
        region_id = self.get(rid).id if rid not in (None, "", 0, "0") else None
        if type not in FLAG_TYPES:
            type = "other"
        entry = {"region": region_id, "type": type, "reason": reason, "agent": "translator"}
        self.flags.append(entry)
        return entry

    def occupied(self, shape, dilate: int = 2, but: Optional[int] = None) -> np.ndarray:
        """0/1 array of pixels covered by the active regions (kept or skipped) other than ``but``,
        so snapping a new region does not take their strokes."""
        canvas = np.zeros(shape[:2], np.uint8)
        polys = [np.round(p).astype(np.int32).reshape(-1, 1, 2) for r in self.active_regions() if r.id != but
                 for p in r.polygons]
        if polys:
            cv2.fillPoly(canvas, polys, 1)
        if dilate > 0:
            canvas = cv2.dilate(canvas, np.ones((2 * dilate + 1,) * 2, np.uint8))
        return canvas

    def overlap(self, bbox, removed: bool = True) -> float:
        """Largest share of ``bbox`` that one active region's box covers (or a box the user deleted)."""
        x1, y1, x2, y2 = bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        best = 0.0
        boxes = [r.bbox for r in self.active_regions()]
        if removed:
            boxes += [item["bbox"] for item in self.removed]
        for rx1, ry1, rx2, ry2 in boxes:
            inter = max(0, min(x2, rx2) - max(x1, rx1)) * max(0, min(y2, ry2) - max(y1, ry1))
            best = max(best, inter / area)
        return best

    def removed_at(self, bbox) -> Optional[dict]:
        """A box the user deleted that covers most of ``bbox``."""
        x1, y1, x2, y2 = bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        for item in self.removed:
            rx1, ry1, rx2, ry2 = item["bbox"]
            inter = max(0, min(x2, rx2) - max(x1, rx1)) * max(0, min(y2, ry2) - max(y1, ry1))
            if inter / area > 0.5:
                return item
        return None

    def add_region(self, snap: SnapResult, direction: str, origin: str = "agent", reason: str = "",
                   hint: str = "", kind: str = "", after=None) -> Region:
        """New region from snapped strokes (text the detector missed)."""
        direction = (direction or "").lower()[:1]
        if direction not in ("v", "h"):
            raise RegionError("direction must be v (vertical) or h (horizontal)")
        gone = self.removed_at(snap.bbox)
        if gone is not None:
            raise RegionError(f"the user deleted the box here in the editor (it read 「{gone.get('text') or ''}」): "
                              "the text stays as they left it, so do not box it again")
        anchor = self.get(after).id if after not in (None, "", 0, "0") else None
        self._checkpoint(f"add_region ({origin})")
        rid = max(self.regions, default=0) + 1
        template = next(iter(self.regions.values())).data if self.regions else {}
        data = {k: copy.deepcopy(template[k]) for k in STYLE_KEYS if k in template}
        x1, y1, x2, y2 = snap.bbox
        data.update({"lines": snap.lines, "texts": [hint] if hint else [], "text": hint, "translation": "",
                     "angle": 0, "font_size": snap.font_size, "fg_colors": snap.fg, "bg_colors": snap.bg,
                     "direction": direction, "alignment": "center",
                     "center": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)]})
        region = Region(id=rid, data=data, ocr_local=hint, origin=origin, origin_kind=kind,
                        erase=(snap.erase_origin[0], snap.erase_origin[1], snap.erase))
        if reason:
            region.reasons.append(reason)
        self.regions[rid] = region
        if anchor is None:
            self.order.append(rid)
        else:
            self.order.insert(self.order.index(anchor) + 1, rid)
        return region

    # --- output -------------------------------------------------------------
    def to_json_regions(self) -> List[dict]:
        """Regions for the JSON. Unfinished regions stay with an empty translation
        so they show up in the editor instead of silently disappearing."""
        out = []
        for region in self.active_regions():
            if region.skipped:
                continue
            if region.unchanged:  # a page checked again: the region as it was, with the editor's fields
                data = copy.deepcopy(region.loaded)
                if region.drop_rich:
                    data.pop("translation_rich", None)
                if region.font and region.font != data.get("font_family"):
                    data["font_family"] = region.font
                out.append(data)
                continue
            data = copy.deepcopy(region.data)
            source = region.source_text if region.source_text else region.ocr_local
            data["text"] = source
            data["texts"] = [source]
            data["translation"] = region.translation or ""
            data["translation_raw"] = region.translation or ""
            data.pop("translation_rich", None)
            if region.loaded_state is not None:  # new text: the box the last render measured is gone
                data.pop("render_box_rect_local", None)
            data["center"] = region.center
            if region.font:
                data["font_family"] = region.font
            out.append(data)
        return out

    def output_mask(self, keep_margin: int = 15, reach: int = 60) -> Optional[str]:
        """Saved mask (base64 PNG) with added regions' strokes put in and
        skipped regions' text taken out. Returns None when nothing changes.

        Mask components that only touch skipped text are cleared. Components
        shared with kept text are cleared only near the skipped text and away
        from the kept text. On an incomplete page the regions the agent did not
        translate are left the same way, so the typeset page keeps their
        original text (they stay in the JSON with an empty translation).
        """
        unfinished = self.status == "incomplete"

        def left(region: Region) -> bool:
            """Its text is taken out of the mask now. On a page checked again, text the earlier run
            skipped is out of the saved mask already, and the mask stays as the user left it there."""
            if region.skipped:
                return not (self.revise and region.was_skipped)
            return unfinished and not region.translation and not region.unchanged

        def kept(region: Region) -> bool:
            return not region.skipped and not left(region)

        skipped = [r for r in self.active_regions() if left(r)]
        # An added region merged into another one keeps its strokes in the
        # mask as long as the region it went into is kept.
        added = [r for r in self.regions.values() if r.erase is not None and kept(self._final(r))]
        encoded = self.page_data.get("mask_raw")
        if not (skipped or added) or not isinstance(encoded, str):
            return None
        mask = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        shape = mask.shape[:2]
        out = mask.copy()
        if added:
            page_w = int(self.page_data.get("original_width") or shape[1])
            page_h = int(self.page_data.get("original_height") or shape[0])
            canvas = np.zeros((page_h, page_w), np.uint8)
            for region in added:
                x, y, patch = region.erase
                h, w = min(patch.shape[0], page_h - y), min(patch.shape[1], page_w - x)
                if h > 0 and w > 0:
                    canvas[y:y + h, x:x + w] |= patch[:h, :w].astype(np.uint8)
            if canvas.shape != shape:
                canvas = cv2.resize(canvas, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            out[canvas > 0] = 255

        if skipped:
            def fill(regions: List[Region]) -> np.ndarray:
                canvas = np.zeros(shape, np.uint8)
                polys = [np.round(p).astype(np.int32).reshape(-1, 1, 2) for r in regions for p in r.polygons]
                if polys:
                    cv2.fillPoly(canvas, polys, 1)
                return canvas

            keep_zone = cv2.dilate(fill([r for r in self.active_regions() if kept(r)]),
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * keep_margin + 1,) * 2))
            skip_seed = fill(skipped)
            skip_touch = cv2.dilate(skip_seed, np.ones((7, 7), np.uint8))
            skip_near = cv2.dilate(skip_seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2))
            _count, labels = cv2.connectedComponents((out > 0).astype(np.uint8), connectivity=8)
            touching_skip = set(np.unique(labels[(skip_touch > 0) & (labels > 0)]).tolist())
            touching_keep = set(np.unique(labels[(keep_zone > 0) & (labels > 0)]).tolist())
            alone = list(touching_skip - touching_keep)
            shared = list(touching_skip & touching_keep)
            if alone:
                out[np.isin(labels, alone)] = 0
            if shared:
                out[np.isin(labels, shared) & (skip_near > 0) & (keep_zone == 0)] = 0
        if np.array_equal(out, mask):
            return None
        ok, png = cv2.imencode(".png", out)
        return base64.b64encode(png.tobytes()).decode("ascii") if ok else None

    def json_data(self) -> dict:
        """The page's JSON as ``save`` writes it."""
        data = dict(self.page_data)
        data["regions"] = self.to_json_regions()
        new_mask = self.output_mask()
        if new_mask is not None:
            data["mask_raw"] = new_mask
        return {self.key: data}

    def save(self) -> None:
        write_json_atomic(self.json_path, self.json_data())

    def sidecar(self) -> dict:
        return {
            "page": os.path.basename(self.image_path),
            "status": self.status,
            "confidence": self.confidence,
            "agent_confidence": self.agent_confidence,
            "check_score": self.check_score,
            "note": self.note,
            "order": self.order,
            "regions": [
                {
                    "id": r.id,
                    "match": {"center": r.center, "bbox": r.bbox},
                    "origin": r.origin,
                    "category": r.category,
                    "ocr_local": r.ocr_local,
                    "ocr_agent": r.source_text,
                    "ruby": r.ruby,
                    "translation": r.translation,
                    "confidence": r.confidence,
                    "skipped": r.skipped,
                    "skip_category": r.skip_category,
                    "merged_into": r.merged_into,
                    "link_group": r.link_group,
                    "reasons": r.reasons,
                    "font": r.font,
                    "style": r.style,
                    "checks": [c["code"] for c in self.checks if c.get("region") == r.id],
                    # a skipped region is not in the JSON: the editor can put it back as it was
                    **({"json": r.data} if r.skipped else {}),
                    # the user's changes, for the next check of the page and the later stages
                    **({"user_edited": True} if r.user_edited else {}),
                    **({"user_font": True} if r.user_font else {}),
                    **({"locked": True} if r.locked else {}),
                }
                for r in (self.regions[i] for i in self.order)
            ],
            "review_flags": self.flags,
            "checks": self.checks,
            **({"removed": self.removed} if self.removed else {}),
            **self.sidecar_extra,
        }
