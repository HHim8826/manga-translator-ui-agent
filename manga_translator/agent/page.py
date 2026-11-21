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
        return self.skipped or bool(self.translation)

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
        self._history.append((label, copy.deepcopy(self.regions), list(self.order), copy.deepcopy(self.flags)))

    def clear_history(self) -> None:
        self._history.clear()

    def undo(self) -> str:
        if not self._history:
            raise RegionError("nothing to undo on this page")
        label, self.regions, self.order, self.flags = self._history.pop()
        return label

    # --- edits --------------------------------------------------------------
    def set_text(self, rid, source_text: str, translation: str, category: str = "dialogue",
                 confidence: Optional[float] = None, ruby: str = "", reason: str = "") -> Region:
        region = self.get(rid)
        translation = with_breaks(translation)
        if not translation.strip():
            raise RegionError(f"#{region.id}: translation is empty; use skip_region for text that stays as it is")
        if category and category not in CATEGORIES:
            raise RegionError(f"category must be one of {', '.join(CATEGORIES)}")
        self._checkpoint(f"set_text #{region.id}")
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
        region = self.get(rid)
        if not reason:
            raise RegionError("reason is required")
        self._checkpoint(f"skip #{region.id}")
        region.skipped = True
        region.skip_category = category or "other"
        region.translation = None
        region.reasons.append(reason)
        return region

    def merge(self, ids, reason: str) -> Region:
        regions = [self.get(i) for i in ids]
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
        region = self.get(rid)
        if not region.translation:
            raise RegionError(f"#{region.id} has no translation yet; call set_text first")
        text = with_breaks(text)
        if "".join(text.replace(BREAK, "").split()) != "".join(region.translation.replace(BREAK, "").split()):
            raise RegionError(f"#{region.id}: the text must be the current translation with line breaks only")
        self._checkpoint(f"line_breaks #{region.id}")
        region.translation = text
        if reason:
            region.reasons.append(reason)
        return region

    def flag(self, rid, type: str, reason: str) -> dict:
        region_id = self.get(rid).id if rid not in (None, "", 0, "0") else None
        if type not in FLAG_TYPES:
            type = "other"
        entry = {"region": region_id, "type": type, "reason": reason, "agent": "translator"}
        self._checkpoint("flag")
        self.flags.append(entry)
        return entry

    def occupied(self, shape, dilate: int = 2) -> np.ndarray:
        """0/1 array of pixels covered by the active regions (kept or skipped),
        so snapping a new region does not take their strokes."""
        canvas = np.zeros(shape[:2], np.uint8)
        polys = [np.round(p).astype(np.int32).reshape(-1, 1, 2) for r in self.active_regions() for p in r.polygons]
        if polys:
            cv2.fillPoly(canvas, polys, 1)
        if dilate > 0:
            canvas = cv2.dilate(canvas, np.ones((2 * dilate + 1,) * 2, np.uint8))
        return canvas

    def overlap(self, bbox) -> float:
        """Largest share of ``bbox`` that one active region's box covers."""
        x1, y1, x2, y2 = bbox
        area = max(1, (x2 - x1) * (y2 - y1))
        best = 0.0
        for region in self.active_regions():
            rx1, ry1, rx2, ry2 = region.bbox
            inter = max(0, min(x2, rx2) - max(x1, rx1)) * max(0, min(y2, ry2) - max(y1, ry1))
            best = max(best, inter / area)
        return best

    def add_region(self, snap: SnapResult, direction: str, origin: str = "agent", reason: str = "",
                   hint: str = "", kind: str = "", after=None) -> Region:
        """New region from snapped strokes (text the detector missed)."""
        direction = (direction or "").lower()[:1]
        if direction not in ("v", "h"):
            raise RegionError("direction must be v (vertical) or h (horizontal)")
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
            data = copy.deepcopy(region.data)
            source = region.source_text if region.source_text else region.ocr_local
            data["text"] = source
            data["texts"] = [source]
            data["translation"] = region.translation or ""
            data["translation_raw"] = region.translation or ""
            data.pop("translation_rich", None)
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
            return region.skipped or (unfinished and not region.translation)

        skipped = [r for r in self.active_regions() if left(r)]
        # An added region merged into another one keeps its strokes in the
        # mask as long as the region it went into is kept.
        added = [r for r in self.regions.values() if r.erase is not None and not left(self._final(r))]
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

            kept = [r for r in self.active_regions() if not left(r)]
            keep_zone = cv2.dilate(fill(kept), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * keep_margin + 1,) * 2))
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

    def save(self) -> None:
        data = dict(self.page_data)
        data["regions"] = self.to_json_regions()
        new_mask = self.output_mask()
        if new_mask is not None:
            data["mask_raw"] = new_mask
        write_json_atomic(self.json_path, {self.key: data})

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
                }
                for r in (self.regions[i] for i in self.order)
            ],
            "review_flags": self.flags,
            "checks": self.checks,
        }
