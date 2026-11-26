"""Tools the agent calls, M2 subset.

Reading: get_candidates, zoom, get_neighbors, local_ocr, measure_bubble
Structure: add_region, merge_regions, split_region, fit_region, skip_region, note_region, set_reading_order,
           link_regions
Text: set_text, set_line_breaks, set_font_style
Memory: glossary_search, glossary_upsert, character_get, character_upsert,
        story_summary_get, story_summary_update, search_past_translations
Flow: flag_for_review, mark_page_done, budget_status, undo

Rules shared by every tool: regions are named by id; the only coordinates the
model gives are rough boxes for text without a region (add_region, zoom), and
the program snaps those to the strokes. Every write takes a ``reason``; bad
arguments return an error text the model can act on instead of raising.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from . import checks, images
from .page import BREAK, CATEGORIES, FLAG_TYPES, PageDoc, Region, RegionError, line_groups, needs_break
from .notes import (COVERS, NEAR, bracketed, covered_text, flat_ground, free_boxes, gap, place_in_box, place_note,
                    usual_size)
from .revise import unskip_erase
from .settings import AgentSettings
from .snap import snap_text
from .store import AgentMemory, normalize
from .font_styles import style_name
from .work_glossary import WORK_CATEGORIES

logger = logging.getLogger("manga_translator.agent")

SKIP_CATEGORIES = ("sign", "sfx", "credit", "not_text", "duplicate", "other")


@dataclass
class ToolResult:
    text: str
    images: List[str] = field(default_factory=list)
    summary: str = ""
    ok: bool = True
    page_done: bool = False


class AgentSession:
    """Everything the tools act on during one run."""

    def __init__(self, settings: AgentSettings, memory: AgentMemory, pages: List[str], events=None,
                 target_lang: str = ""):
        self.settings = settings
        self.target_lang = target_lang
        self.memory = memory
        self.pages = pages
        self.events = events
        self.page_index = -1
        self.doc: Optional[PageDoc] = None
        self.page_image = None
        self.page_array: Optional[np.ndarray] = None
        self.last_view: Optional[Dict[str, Any]] = None
        self.past: List[Dict[str, Any]] = []
        self.last_prompt_tokens = 0
        self.page_steps = 0
        self.work_terms: List[Any] = []
        """Glossary terms of the work's prompt file (``work_prompt.WorkTerm``), for the checks."""
        self.font_choices: List[str] = []
        """Fonts on the font sheet (font_family values); set_font_style font n picks font_choices[n - 1]."""
        self.styles_used: Dict[str, set] = {}
        """Font style → pages of this run with a region in it (a used style keeps its font)."""
        self.line_breaks = False
        """AI line breaks (render setting disable_auto_wrap): the renderer breaks lines only at [BR], so
        translations of multi-line sources need them."""
        self._reset_page_state()

    def _reset_page_state(self) -> None:
        self.zoomed_ids: set = set()
        self.zoom_boxes: List[List[float]] = []
        self.review_shown = False
        self.recall_failed = False
        self.page_tool_errors = 0
        self.render_issues: List[checks.Issue] = []
        """``render_check`` issues of the page, measured by the runner before each mark_page_done."""

    def was_zoomed(self, region: Region) -> bool:
        """Zoomed by id (also before a merge into it), or inside a zoomed area."""
        if any(self.doc._final(self.doc.regions[i]).id == region.id for i in self.zoomed_ids if i in self.doc.regions):
            return True
        cx, cy = region.center
        return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in self.zoom_boxes)

    @property
    def page_name(self) -> str:
        return self.doc.stem if self.doc else ""

    def start_page(self, index: int, doc: PageDoc) -> None:
        self.page_index = index
        self.doc = doc
        self.page_image = images.open_page(doc.image_path)
        self.page_array = np.asarray(self.page_image)
        self.last_view = None
        self.page_steps = 0
        self._reset_page_state()

    def finish_page(self) -> None:
        if self.doc is None:
            return
        for region in self.doc.active_regions():
            if region.translation:
                self.past.append({"page": self.doc.stem, "id": region.id, "source": region.source_text or region.ocr_local,
                                  "translation": region.translation, "category": region.category})

    def marks(self):
        out = []
        for region in self.doc.active_regions():
            state = "skipped" if region.skipped else ("done" if region.translation else "pending")
            out.append((region.id, region.bbox, state))
        return out

    def candidate_lines(self) -> str:
        lines = []
        revise = self.doc.revise
        for region in self.doc.active_regions():
            x1, y1, x2, y2 = region.bbox
            local = region.ocr_local.replace("\n", " / ") or "(empty)"
            if revise and region.loaded_state is not None and region.user_edited and not region.ocr_local:
                source = "drawn by the user in the editor"
            elif revise and region.loaded_state is not None and region.unchanged and region.source_text:
                source = f"earlier reading: 「{_short(region.source_text, 80)}」"
            elif region.origin == "recall":
                source = f"added by the recall check ({region.origin_kind or 'text'}), its reading (hint): 「{local}」"
            elif region.origin == "agent":
                source = "added by you with add_region"
            elif region.origin == "cut":
                source = "cut out of a detected line with split_region"
            elif region.note_of is not None:
                source = f"the translation of #{region.note_of} set beside it (note_region)"
            else:
                source = f"local OCR (weak hint): 「{local}」"
            count = ""
            if self.line_breaks:
                room, rows = checks.room_counts(region)
                count = f"{region.line_count} source line(s), fits about {room} characters per line and {rows} line(s), "
            status = region.status()
            notes = self.doc.notes_of(region)
            if notes:
                status += f", translated beside it in #{notes[0].id}"
            if revise and region.translation and not region.skipped:
                status += f" 「{region.translation}」"
                if region.style:
                    status += f" [style {region.style}]"
            if revise and region.locked:
                status += " — LOCKED: the user changed it in the editor; leave it"
            lines.append(f"#{region.id}: {region.direction}, {x2 - x1}x{y2 - y1}px, font≈{region.font_size}px, "
                         f"{count}{source}, {status}{_apart(region, revise)}")
        return "\n".join(lines) if lines else "(no candidate regions on this page)"


def _apart(region: Region, revise: bool = False) -> str:
    """For a pending region whose lines stand in groups apart from each other: the groups, so the
    model can split two bubbles (or an aside beside the main text) that were boxed as one text. On a
    page checked again (``revise``) translated regions get it too: their one translation may cover
    both parts, or only one."""
    if region.skipped or region.locked or (region.translation and not revise):
        return ""
    groups = line_groups(region)
    if len(groups) < 2:
        return ""
    polys = region.polygons
    parts = []
    for group in groups:
        points = np.concatenate([polys[i] for i in group])
        box = ", ".join(str(int(v)) for v in (points[:, 0].min(), points[:, 1].min(),
                                                 points[:, 0].max(), points[:, 1].max()))
        parts.append(f"[{box}] ({len(group)} line{'s' if len(group) > 1 else ''})")
    return (f"; its lines stand apart in {len(groups)} groups: {', '.join(parts)}. Two bubbles, the two parts "
            "of a joined bubble, or an aside beside the main text: split_region them (box = a group above) and "
            "translate each part in its place; one block of text, such as a stepped narration box, stays one region")


Handler = Callable[[AgentSession, Dict[str, Any]], ToolResult]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Handler
    writes: bool = False


def _obj(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


ID = {"type": "integer", "description": "Region number as drawn on the page image"}
IDS = {"type": "array", "items": {"type": "integer"}, "description": "Region numbers"}
REASON = {"type": "string", "description": "One short sentence: why. Shown to the human on the Agents page"}
BOX = {"type": "array", "items": {"type": "number"},
       "description": "[x1, y1, x2, y2] in pixels of the page image (size in the page message); a rough box is fine"}


def _short(text: str, limit: int = 40) -> str:
    text = (text or "").replace(BREAK, " / ").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- reading ------------------------------------------------------------------
def get_candidates(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    url, shown = images.page_url(images.draw_marks(s.page_image, s.marks()), s.settings.page_image_max_side)
    return ToolResult(
        text=(f"Page {s.page_name} ({s.page_image.width}x{s.page_image.height} px{', ' + shown if shown else ''}): "
              f"red = pending, green = translated, gray = skipped.\n{s.candidate_lines()}"),
        images=[url], summary=f"{len(s.doc.active_regions())} regions")


def _page_box(s: AgentSession, raw: Any) -> List[float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise RegionError("box must be [x1, y1, x2, y2] in page pixels")
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
    except (TypeError, ValueError):
        raise RegionError("box must be four numbers [x1, y1, x2, y2]")
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    w, h = s.page_image.size
    x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise RegionError(f"box {list(raw)} is empty or outside the page ({w}x{h} px)")
    return [x1, y1, x2, y2]


CELL = re.compile(r"\s*([A-Ha-h])\s*([1-8])\s*(?:[:\-]\s*([A-Ha-h])\s*([1-8])\s*)?")


def _cells_box(s: AgentSession, cells: str) -> List[float]:
    if not s.last_view:
        raise RegionError("cells refer to the grid of your last zoom on this page; call zoom first")
    match = CELL.fullmatch(str(cells))
    if not match:
        raise RegionError("cells must look like C4 or B3:D5 (grid of the last zoom)")
    columns = images.GRID_COLUMNS
    c1, r1 = columns.index(match[1].upper()), int(match[2]) - 1
    c2, r2 = (columns.index(match[3].upper()), int(match[4]) - 1) if match[3] else (c1, r1)
    c1, c2 = sorted((c1, c2))
    r1, r2 = sorted((r1, r2))
    left, top, right, bottom = s.last_view["crop"]
    cw, ch = (right - left) / len(columns), (bottom - top) / images.GRID_ROWS
    return [left + c1 * cw, top + r1 * ch, left + (c2 + 1) * cw, top + (r2 + 1) * ch]


def zoom(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    if a.get("id") not in (None, ""):
        region = s.doc.get(a["id"])
        box, label, skip_id = region.bbox, f"#{region.id}", region.id
        s.zoomed_ids.add(region.id)
    elif a.get("box") is not None:
        box = _page_box(s, a["box"])
        label, skip_id = "area [" + ", ".join(str(int(v)) for v in box) + "]", None
        s.zoom_boxes.append(box)
    else:
        raise RegionError("give id (a region) or box ([x1, y1, x2, y2] page pixels, any area of the page)")
    others = [poly.tolist() for r in s.doc.active_regions() if r.id != skip_id for poly in r.polygons]
    image, info = images.zoom(s.page_image, box, others, scale=float(a.get("scale") or 3),
                              mask_others=a.get("mask_others", True) is not False)
    s.last_view = info
    cell = info.get("grid", {}).get("cell_page_px")
    return ToolResult(
        text=(f"Zoom of {label} at {info['scale']}x; other regions' text is grayed out. "
              f"Grid A1-H8, each cell ≈ {cell[0]}x{cell[1]} page px; add_region accepts these cells."
              if cell else f"Zoom of {label}."),
        images=[images.data_url(image)], summary=f"{label} ×{info['scale']}")


def get_neighbors(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    span = max(1, min(2, int(a.get("range") or 1)))
    parts, labels = [], []
    for offset in range(-span, span + 1):
        index = s.page_index + offset
        if offset == 0 or not 0 <= index < len(s.pages):
            continue
        parts.append(images.thumbnail(s.pages[index], 640))
        labels.append(f"{'previous' if offset < 0 else 'next'} page ({offset:+d})")
    if not parts:
        return ToolResult(text="This page has no neighbors in this run.", summary="none")
    sheet = images.side_by_side(parts, labels)
    return ToolResult(text="Thumbnails: " + ", ".join(labels), images=[images.data_url(sheet, max_side=1600)],
                      summary=", ".join(labels))


def local_ocr(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc.get(a["id"])
    prob = region.data.get("prob")
    return ToolResult(text=json.dumps({"id": region.id, "local_ocr": region.ocr_local, "detector_prob": prob},
                                      ensure_ascii=False),
                      summary=f"#{region.id}: {_short(region.ocr_local)}")


def measure_bubble(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc.get(a["id"])
    x1, y1, x2, y2 = region.bbox
    w, h = x2 - x1, y2 - y1
    fs = max(8, region.font_size or int(min(w, h) * 0.8))
    if region.direction.startswith("v"):
        cols, per = max(1, int(w // (fs * 1.1))), max(1, int(h // fs))
        fit = f"about {cols} column(s) × {per} characters at the original size"
    else:
        rows, per = max(1, int(h // (fs * 1.2))), max(1, int(w // fs))
        fit = f"about {rows} line(s) × {per} characters at the original size"
    return ToolResult(
        text=(f"#{region.id}: text box {w}x{h}px, {region.direction}, original font ≈{fs}px, {fit}. "
              "The renderer can grow into the surrounding bubble and shrink the font, so this is a guide, not a limit."),
        summary=f"#{region.id} {w}x{h}")


# --- structure ----------------------------------------------------------------
def add_region(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    if a.get("cells"):
        box = _cells_box(s, a["cells"])
    elif a.get("box") is not None:
        box = _page_box(s, a["box"])
    else:
        raise RegionError("give box ([x1, y1, x2, y2] page pixels) or cells (grid of your last zoom)")
    direction = str(a.get("direction") or "")
    snap = snap_text(s.page_array, box, direction or "v", exclude=s.doc.occupied(s.page_array.shape),
                     grow=not a.get("exact"))
    if snap is None:
        x1, y1, x2, y2 = box
        inside = [r.id for r in s.doc.active_regions()
                  if min(x2, r.bbox[2]) - max(x1, r.bbox[0]) > 0 and min(y2, r.bbox[3]) - max(y1, r.bbox[1]) > 0]
        hint = (f" The strokes there belong to #{inside[0]}: to make a separate region of part of its text (a "
                f"gloss or a name on the same detected line), split_region #{inside[0]} with this box."
                if len(inside) == 1 else "")
        raise RegionError("no text strokes found there (strokes inside other regions do not count)." + hint +
                          " Otherwise zoom on the area and give a tighter box, or grid cells.")
    region = s.doc.add_region(snap, direction, origin="agent", reason=a.get("reason", ""), after=a.get("after"))
    others = [poly.tolist() for r in s.doc.active_regions() if r.id != region.id for poly in r.polygons]
    preview = images.snap_preview(s.page_image, snap.bbox, snap.lines, snap.erase, snap.erase_origin, others)
    x1, y1, x2, y2 = snap.bbox
    warning = f" Warning: {snap.warning}." if snap.warning else ""
    exact = "" if a.get("exact") else ", or exact: true when it caught strokes of a neighbouring label"
    return ToolResult(
        text=(f"Added #{region.id}: {x2 - x1}x{y2 - y1}px at [{x1}, {y1}, {x2}, {y2}], {len(snap.lines)} line(s)"
              f"{_slant_note(snap)}, font≈{snap.font_size}px. Red pixels will be erased, the green box is where the "
              f"translation goes.{warning} If it caught the wrong strokes or missed some, call undo and try a "
              f"tighter or larger box{exact}. Then set_text or skip_region #{region.id} like any region."),
        images=[images.data_url(preview)], summary=f"#{region.id} {x2 - x1}x{y2 - y1}")


def _slant_note(snap) -> str:
    if not snap.angle:
        return ""
    way = "rising" if snap.angle < 0 else "falling"
    return f", written on a slant ({way} {abs(snap.angle):.0f}° to the right; the translation is drawn along it)"


def merge_regions(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    kept = s.doc.merge(a["ids"], a.get("reason", ""))
    return ToolResult(text=f"Merged into #{kept.id}. Set its text again with set_text.",
                      summary="+".join(f"#{i}" for i in a["ids"]) + f" → #{kept.id}")


def _box_arg(s: AgentSession, a: Dict[str, Any]) -> Optional[List[float]]:
    if a.get("cells"):
        return _cells_box(s, a["cells"])
    if a.get("box") is not None:
        return _page_box(s, a["box"])
    return None


def _direction(a: Dict[str, Any], region: Region, box) -> str:
    """The direction asked for, else the region's, else the box's shape."""
    given = str(a.get("direction") or "").lower()[:1]
    if given in ("v", "h"):
        return given
    if region.direction[:1] in ("v", "h"):
        return region.direction[:1]
    return "h" if box[2] - box[0] >= box[3] - box[1] else "v"


def _own_strokes(s: AgentSession, region: Region, box, direction: str):
    """``snap_text`` of the strokes of ``region`` whose middle lies in ``box``: its lines (with a
    little room around them) and nothing that belongs to another region."""
    shape = s.page_array.shape
    own = np.zeros(shape[:2], np.uint8)
    polys = [np.round(p).astype(np.int32).reshape(-1, 1, 2) for p in region.polygons]
    if polys:
        cv2.fillPoly(own, polys, 1)
    pad = max(3, int(round(0.25 * (region.font_size or 16))))
    own = cv2.dilate(own, np.ones((2 * pad + 1,) * 2, np.uint8))
    exclude = ((own == 0) | (s.doc.occupied(shape, but=region.id) > 0)).astype(np.uint8)
    return snap_text(s.page_array, box, direction, exclude=exclude, grow=False)


def _gray(s: AgentSession) -> np.ndarray:
    page = s.page_array
    return cv2.cvtColor(np.ascontiguousarray(page[:, :, :3]), cv2.COLOR_RGB2GRAY) if page.ndim == 3 else page


def split_region(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    box = _box_arg(s, a)
    if box is None:
        raise RegionError("give box ([x1, y1, x2, y2] page pixels) or cells (grid of your last zoom) around the "
                          "lines of the new region, or around the strokes of a part inside a line")
    region = s.doc._plain(s.doc.get(a["id"]), "split")
    inside = s.doc.lines_in(region.id, box)
    if 0 < len(inside) < region.line_count:  # whole lines move
        new = s.doc.split(region.id, box, a.get("reason", ""))
        old = s.doc.get(region.id)
        return ToolResult(
            text=(f"#{old.id} keeps {old.line_count} line(s) at {old.bbox}; its other {new.line_count} line(s) are "
                  f"now #{new.id} at {new.bbox}. Both need set_text or skip_region (any text #{old.id} had was "
                  f"cleared)."),
            summary=f"#{old.id} → #{old.id} + #{new.id}")
    # a part inside a line: the strokes in the box are cut out of it
    direction = _direction(a, region, box)
    snap = _own_strokes(s, region, box, direction)
    try:
        if snap is None:
            raise RegionError("no strokes")
        new = s.doc.cut(region.id, snap, direction, a.get("reason", ""), gray=_gray(s))
    except RegionError as error:
        if snap is None or region.line_count >= 2:
            s.doc.split(region.id, box, a.get("reason", ""))  # raises: says which lines a box can take
        raise error
    old = s.doc.get(region.id)
    others = [poly.tolist() for r in s.doc.active_regions() if r.id != new.id for poly in r.polygons]
    preview = images.snap_preview(s.page_image, snap.bbox, snap.lines, snap.erase, snap.erase_origin, others)
    x1, y1, x2, y2 = snap.bbox
    warning = f" Warning: {snap.warning}." if snap.warning else ""
    return ToolResult(
        text=(f"Cut out of #{old.id}: the {snap.strokes} stroke(s) inside the box are now #{new.id}, {direction}, "
              f"{len(snap.lines)} line(s){_slant_note(snap)} at [{x1}, {y1}, {x2}, {y2}], font≈{snap.font_size}px "
              f"(red in the image); "
              f"#{old.id} keeps the rest at {old.bbox} (gray).{warning} If the wrong strokes moved, undo and give a "
              f"tighter box (zoom first). Both need set_text or skip_region (any text #{old.id} had was cleared): "
              f"the strokes of a skipped part stay drawn, those of a translated part are erased."),
        images=[images.data_url(preview)], summary=f"#{old.id} → #{old.id} + #{new.id} (cut)")


def fit_region(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc._plain(s.doc.get(a["id"]), "fit")
    box = _box_arg(s, a) or region.bbox
    direction = _direction(a, region, box)
    exclude = s.doc.occupied(s.page_array.shape, but=region.id)
    snap = snap_text(s.page_array, box, direction, exclude=exclude, grow=False)
    if snap is None:
        raise RegionError(f"no text strokes found in the box (strokes of other regions do not count). Zoom on "
                          f"#{region.id} and give a box around its text.")
    before = f"{region.line_count} line(s) at {region.bbox}, font≈{region.font_size}px"
    s.doc.fit(region.id, snap, a.get("reason", ""), direction=direction)
    others = [poly.tolist() for r in s.doc.active_regions() if r.id != region.id for poly in r.polygons]
    preview = images.snap_preview(s.page_image, snap.bbox, snap.lines, snap.erase, snap.erase_origin, others)
    x1, y1, x2, y2 = snap.bbox
    warning = f" Warning: {snap.warning}." if snap.warning else ""
    breaks = ""
    if region.translation and s.line_breaks:
        breaks = f" Check the line breaks of its translation against {len(snap.lines)} source line(s)."
    return ToolResult(
        text=(f"#{region.id} now has {len(snap.lines)} line(s){_slant_note(snap)} at [{x1}, {y1}, {x2}, {y2}], "
              f"{direction}, "
              f"font≈{snap.font_size}px (was {before}). Red pixels will be erased, the green box is where the "
              f"translation goes.{warning} If it caught the wrong strokes, undo and give a tighter or larger box. "
              f"Its text stays.{breaks}"),
        images=[images.data_url(preview)], summary=f"#{region.id} {x2 - x1}x{y2 - y1} font {snap.font_size}")


def skip_region(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    category = a.get("category") or "other"
    if category not in SKIP_CATEGORIES:
        raise RegionError(f"category must be one of {', '.join(SKIP_CATEGORIES)}")
    region = s.doc.skip(a["id"], category, a.get("reason", ""))
    return ToolResult(text=f"#{region.id} skipped ({category}): it stays as in the original and is not erased.",
                      summary=f"#{region.id} ({category})")


def _on_flat_ground(s: AgentSession, region: Region) -> bool:
    """``notes.flat_ground`` of the region's lines, or of its strokes snapped as fit_region would: the
    detector's loose box round 「北」 on the ch42 map takes in the circle round it."""
    if flat_ground(s.page_array, region.polygons, font_size=region.font_size):
        return True
    snap = snap_text(s.page_array, region.bbox, _direction({}, region, region.bbox),
                     exclude=s.doc.occupied(s.page_array.shape, but=region.id), grow=False)
    return snap is not None and flat_ground(s.page_array, [np.asarray(q, dtype=np.float64) for q in snap.lines],
                                            font_size=snap.font_size)


def note_region(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc.get(a["id"])
    text = bracketed(a.get("translation", ""))
    if _on_flat_ground(s, region):
        raise RegionError(f"#{region.id} is on flat ground (plain paper or one flat color round it, as in a "
                          f"bubble, a box or a circle; an outline that only touches it is drawn again): erasing it "
                          f"leaves the drawing whole, so translate it in its place with set_text. note_region is only for text over art "
                          f"that erasing would break: lines that run under or into it, hatching, detailed drawing.")
    x1, y1, x2, y2 = region.bbox
    own = region.direction[:1] if region.direction[:1] in ("v", "h") else ("v" if y2 - y1 > x2 - x1 else "h")
    direction = str(a.get("direction") or "").lower()[:1] or own
    if direction not in ("v", "h"):
        raise RegionError("direction must be v (vertical) or h (horizontal)")
    source_px = region.font_size or min(x2 - x1, y2 - y1)
    sized = [r for r in s.doc.regions.values() if r.font_size and r.merged_into is None and r.note_of is None]
    page_px = usual_size([r.font_size for r in sized if r.category in ("dialogue", "narration")],
                         [r.font_size for r in sized])
    box = _box_arg(s, a)
    if box is not None:
        place = place_in_box(_gray(s), box, direction, source_px, text, page_px)
        if place is None:
            raise RegionError(f"at its size the note does not fit on the page in any of 1-4 lines: leave box out, "
                              f"or translate #{region.id} shorter. The note placed before stays.")
        # the page's text, the region's own too, but not its notes (the one moved goes)
        notes = {n.id for n in s.doc.notes_of(region)}
        texts = {r.id: r.polygons for r in s.doc.active_regions() if r.id not in notes}
        over, share = covered_text(place.lines, texts)
        if share > COVERS:
            whose = "its own text" if over == region.id else f"the text of #{over}"
            spots = free_boxes(_gray(s), box, direction, source_px, text, page_px, texts, region.polygons,
                               NEAR * place.font_size)
            if spots:
                hint = ("The nearest boxes that keep it off text, in page pixels: "
                        + ", ".join(f"{b} ({moved} px {way})" for way, moved, b in spots)
                        + ". Give one of them, another empty spot right beside the text, or leave box out.")
            else:
                hint = "No box near it keeps the note off text: give an empty spot on another side of the text, " \
                       "or leave box out."
            raise RegionError(f"that spot puts the note over {whose} ({int(round(100 * share))}% of it): a note sits "
                              f"on an empty spot right beside its text. {hint} The note placed before stays.")
        away, near = gap(place.lines, region.polygons), NEAR * place.font_size
        if away > near:
            raise RegionError(f"that spot puts the note {away:.0f} px from #{region.id}: a note sits right beside "
                              f"its text, within {near:.0f} px, so the reader sees which text it belongs to. Give a "
                              f"spot that close, or leave box out: lines, screentone or a panel border under the "
                              f"note are fine.")
    else:
        try:
            angle = float(region.data.get("angle") or 0.0)
        except (TypeError, ValueError):
            angle = 0.0
        # the region's own earlier note goes, so its place is free
        blocked = s.doc.occupied(s.page_array.shape, dilate=3,
                                 but={region.id, *(n.id for n in s.doc.notes_of(region))})
        place = place_note(_gray(s), region.polygons, direction, angle if direction == own else 0.0, source_px,
                           text, blocked, page_px)
        if place is None:
            raise RegionError(f"there is no room on the page beside #{region.id}: give box, an empty spot near it "
                              f"in page pixels, or cells of your last zoom")
    note = s.doc.add_note(region.id, place, a.get("source_text", ""), a.get("category") or "other",
                      a.get("confidence"), a.get("reason", ""))
    others = [poly.tolist() for r in s.doc.active_regions() if r.id not in (region.id, note.id) for poly in r.polygons]
    preview = images.note_preview(s.page_image, [p.tolist() for p in region.polygons], place.lines, place.text,
                                  place.font_size, others)
    nx1, ny1, nx2, ny2 = place.bbox
    where = "centered on the box you gave" if place.side == "box" else f"{place.side} of the text"
    if place.outgrows:
        where += f" (larger than the box: {nx2 - nx1}x{ny2 - ny1} px is what it needs to be read)"
    lines = place.text.count(BREAK) + 1
    warning = ""
    if place.crowded:
        warning = (" Warning: there is no free spot beside it, so the note covers other text: give box, an empty "
                   "spot near it.")
    return ToolResult(
        text=(f"#{region.id} stays as drawn (green); its translation 「{place.text}」 is #{note.id}, {place.direction}, "
              f"{lines} line(s), font {place.font_size}px at [{nx1}, {ny1}, {nx2}, {ny2}], {where} (orange box), "
              f"over {int(round(100 * place.ink))}% ink.{warning} Lines, screentone and panel borders under it are "
              f"fine. Only if it covers a face or other text, note_region #{region.id} again with box: an empty spot "
              f"right beside the text (within {NEAR * place.font_size:.0f} px of it), in page pixels or cells of "
              f"your last zoom. undo takes the note back."),
        images=[images.data_url(preview)], summary=f"#{region.id} → #{note.id}: {_short(place.text, 30)}")


def set_reading_order(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    order = s.doc.set_order(a["ids"])
    return ToolResult(text="Reading order: " + ", ".join(f"#{i}" for i in order if s.doc.regions[i].active),
                      summary=" → ".join(f"#{i}" for i in a["ids"]))


def link_regions(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    group = s.doc.link(a["ids"], a.get("reason", ""))
    return ToolResult(text=f"Linked as sentence group {group}. Translate them so the parts read as one sentence.",
                      summary="+".join(f"#{i}" for i in a["ids"]))


# --- text ---------------------------------------------------------------------
def _font_choice(s: AgentSession, value: Any) -> Optional[str]:
    """None leaves the region's font as it is; 0 is the user's font."""
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise RegionError("font must be a number from the font sheet (0 = the user's font)")
    if number == 0:
        return ""
    if not s.font_choices:
        raise RegionError("this run has no font sheet; leave font out")
    if not 1 <= number <= len(s.font_choices):
        raise RegionError(f"there is no font {number}; the font sheet lists 0-{len(s.font_choices)}")
    return s.font_choices[number - 1]


NO_STYLE = ("none", "default")


def _style_choice(s: AgentSession, value: Any, category: str, reason: str) -> Optional[tuple]:
    """(style, font) for set_text; None leaves the region's style as it is, ("", "") is the user's font."""
    if value is None or str(value).strip() == "":
        return None
    name = style_name(value)
    if name in NO_STYLE:
        return "", ""
    if not s.font_choices:
        raise RegionError("this run has no font styles; leave style out")
    styles = s.memory.font_styles
    if name not in styles:
        raise RegionError(f"there is no font style {name!r} (styles: {', '.join(styles) or 'none yet'}); "
                          "define it with set_font_style first")
    if category == "dialogue" and not str(reason or "").strip():
        raise RegionError("normal dialogue keeps the user's font: give dialogue a style only when its lettering "
                          "clearly differs from the dialogue around it, and say how in reason")
    return name, styles[name]["font"]


def set_font_style(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    if not s.font_choices:
        raise RegionError("this run has no font sheet, so there are no font styles")
    name = style_name(a.get("name"))
    if name in NO_STYLE:
        raise RegionError(f"{name!r} means the user's font; pick another name")
    font = _font_choice(s, a.get("font"))
    if not font:
        raise RegionError(f"give the number of a font on the font sheet (1-{len(s.font_choices)}); text without a "
                          "style already uses the user's font")
    reason = str(a.get("reason") or "").strip()
    if not reason:
        raise RegionError("reason is required: what the original lettering of this kind of text looks like")
    old = s.memory.font_styles.get(name)
    used = sorted(s.styles_used.get(name, ()))
    if old and old.get("font") != font and used:
        raise RegionError(f"style {name!r} already sets {old['font']} on page(s) {', '.join(used)}; keep it so the "
                          "chapter stays consistent, or define another style for a different lettering")
    status = s.memory.font_style_set(name, font, reason, s.page_name)
    return ToolResult(text=(f"Font style {name!r} = {font} ({status}). Give style \"{name}\" in set_text for every "
                            "region with this lettering."),
                      summary=f"{name} → {font.replace('::', ' ')} ({status})")


def set_text(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    category = a.get("category") or "dialogue"
    style = _style_choice(s, a.get("style"), category, a.get("reason", ""))
    region = s.doc.set_text(a["id"], a.get("source_text", ""), a.get("translation", ""),
                            category, a.get("confidence"), a.get("ruby", ""), a.get("reason", ""))
    if s.doc.revise:  # text the earlier run skipped: find its strokes so the render erases them
        unskip_erase(s.doc, region, s.page_array)
    if style is not None:
        region.style, region.font = style
        if region.style:
            s.styles_used.setdefault(region.style, set()).add(s.page_name)
    remaining = len(s.doc.missing())
    summary = f"#{region.id}: {_short(region.source_text, 24)} → {_short(region.translation, 30)}"
    if region.style:
        summary += f" [{region.style}]"
    text = f"#{region.id} saved. {remaining} region(s) on this page still need a result."
    if s.line_breaks and needs_break(region.line_count, region.translation):
        text += (f" Its source has {region.line_count} lines but the translation has no {BREAK}, so it would be "
                 f"typeset as one line: add {BREAK} at natural breaks with set_line_breaks.")
    else:
        text += _overflow_note(s, region)
    # said now, not at mark_page_done: a refused mark_page_done costs a request, which resends the page
    for issue in checks.region_issues(s, region):
        if issue.code == "needs_second_read":
            text += " " + checks.agent_text(issue) + " Do it before mark_page_done."
    return ToolResult(text=text, summary=summary)


def _overflow_note(s: AgentSession, region: Region) -> str:
    """With AI line breaks: what to fix when the lines run out of the box (``checks.line_overflow``)."""
    over = checks.line_overflow(region, region.translation) if s.line_breaks else None
    if not over:
        return ""
    return " " + checks.agent_text(checks.Issue("line_overflow", region.id, over, 0.0))


def set_line_breaks(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc.set_line_breaks(a["id"], a.get("text", ""), a.get("reason", ""))
    return ToolResult(text=f"#{region.id} line breaks saved.{_overflow_note(s, region)}",
                      summary=f"#{region.id}: {_short(region.translation)}")


# --- memory -------------------------------------------------------------------
def glossary_search(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    q = normalize(a.get("query", ""))
    work = [{"source": t.source, "target": " / ".join(t.targets), "note": t.condition, "from": "work prompt file"}
            for t in s.work_terms
            if not q or q in normalize(t.source) or any(q in normalize(x) for x in t.targets)]
    hits = work[:30] + s.memory.glossary_search(a.get("query", ""))
    return ToolResult(text=json.dumps(hits, ensure_ascii=False) if hits else "No glossary entries match.",
                      summary=f"{_short(a.get('query', ''), 20)}: {len(hits)}")


def glossary_upsert(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    category = a.get("category", "") if a.get("category") in WORK_CATEGORIES else ""
    status = s.memory.glossary_upsert(a["source"], a["target"], a.get("note", ""), s.page_name, category)
    return ToolResult(text=f"Glossary entry {status}.", summary=f"{a['source']} → {a['target']} ({status})")


def character_get(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    found = s.memory.character_get(a.get("name", ""))
    return ToolResult(text=json.dumps(found, ensure_ascii=False) if found else "No matching character.",
                      summary=f"{_short(a.get('name', '') or 'all', 20)}: {len(found)}")


def character_upsert(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    fields = {k: a.get(k) for k in ("target_name", "reading", "gender", "speech_style", "notes")}
    if a.get("category") in ("Person", "Creature"):
        fields["category"] = a["category"]
    status = s.memory.character_upsert(a["name"], fields, s.page_name)
    return ToolResult(text=f"Character {status}.", summary=f"{a['name']} ({status})")


def story_summary_get(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    return ToolResult(text=s.memory.story or "(no summary yet)", summary=f"{len(s.memory.story)} chars")


def story_summary_update(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    s.memory.story_update(a["summary"])
    return ToolResult(text="Story summary saved.", summary=f"{len(s.memory.story)} chars")


def search_past_translations(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    q = normalize(a.get("query", ""))
    if not q:
        raise RegionError("query is required")
    hits = [p for p in s.past if q in normalize(p["source"]) or q in normalize(p["translation"])][:20]
    return ToolResult(text=json.dumps(hits, ensure_ascii=False) if hits else "No earlier translation matches.",
                      summary=f"{_short(a.get('query', ''), 20)}: {len(hits)}")


# --- flow ---------------------------------------------------------------------
def flag_for_review(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    flag_type = a.get("type") or "other"
    if flag_type not in FLAG_TYPES:
        raise RegionError(f"type must be one of {', '.join(FLAG_TYPES)}")
    entry = s.doc.flag(a.get("id"), flag_type, a.get("reason", ""))
    if s.events:
        s.events.emit("flag", page=s.page_name, region=entry["region"], flag=flag_type, reason=entry["reason"])
    return ToolResult(text="Added to the review list. Keep going.",
                      summary=(f"#{entry['region']} " if entry["region"] else "") + flag_type)


def mark_page_done(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    missing = s.doc.missing()
    if missing:
        ids = ", ".join(f"#{i}" for i in missing)
        return ToolResult(text=(f"Not finished: {ids} have neither a translation nor a skip. "
                                "Call set_text or skip_region for each, then mark_page_done again."),
                          ok=False, summary=f"missing {ids}")
    agent_confidence = max(0.0, min(1.0, float(a.get("confidence", 0))))
    issues = checks.page_issues(s)
    lang = checks.human_language(s.target_lang)
    if issues and not s.review_shown:
        # One round of program checks: the model fixes what is wrong, then finishes.
        s.review_shown = True
        listed = "\n".join("- " + checks.agent_text(i) for i in issues)
        return ToolResult(
            text=("Not finished yet. The program checks found:\n" + listed + "\nFix what is really wrong "
                  "(zoom, set_text, ...); leave what is fine as it is. Then call mark_page_done again: the second "
                  "call finishes the page, and anything still listed lowers its confidence."),
            ok=False, summary=checks.codes(issues))
    score = checks.page_score(s, issues)
    doc = s.doc
    doc.status = "done"
    doc.agent_confidence = agent_confidence
    doc.check_score = score
    doc.confidence = round(min(score, agent_confidence), 3)
    doc.checks = [i.to_dict() for i in issues]
    doc.note = a.get("note", "")

    def add_flag(region, flag_type, reason):
        entry = doc.flag(region, flag_type, reason)
        if s.events:
            s.events.emit("flag", page=s.page_name, region=entry["region"], flag=flag_type, reason=reason,
                          source="check")

    for issue in issues:
        if issue.code in checks.SERIOUS:
            add_flag(issue.region, checks.SERIOUS[issue.code], checks.human_text(lang, issue.code, **issue.detail))
    if doc.confidence < s.settings.low_confidence_threshold and not doc.flags:
        add_flag(None, "low_confidence", a.get("note") or checks.human_text(
            lang, "page", score=doc.confidence, summary=checks.summary(lang, issues)))
    return ToolResult(text=(f"Page done. Confidence {doc.confidence:.2f} (program checks {score:.2f}, "
                            f"yours {agent_confidence:.2f})."),
                      summary=f"confidence {doc.confidence:.2f}", page_done=True)


def budget_status(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    st = s.settings
    return ToolResult(text=json.dumps({
        "context_tokens": s.last_prompt_tokens, "compact_at": st.compact_threshold_tokens,
        "hard_limit": st.max_context_tokens, "steps_this_page": s.page_steps,
        "max_steps_per_page": st.max_steps_per_page, "page": f"{s.page_index + 1}/{len(s.pages)}"}),
        summary=f"{s.last_prompt_tokens} tokens")


def undo(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    label = s.doc.undo()
    return ToolResult(text=f"Undone: {label}.", summary=label)


TOOLS: List[Tool] = [
    Tool("get_candidates", "Numbered page image and the candidate list with current status. Call again to see progress.",
         _obj({}, []), get_candidates),
    Tool("zoom", "Magnified crop with an A1-H8 grid; other regions' text is grayed out. Give id for a region "
         "(small, handwritten, stylized or uncertain text: second read) or box for any area of the page "
         "(to look at text without a region before add_region).",
         _obj({"id": ID, "box": BOX, "scale": {"type": "number", "description": "1-6, default 3"},
               "mask_others": {"type": "boolean", "description": "Gray out other regions (default true)"}}, []),
         zoom),
    Tool("get_neighbors", "Thumbnails of the previous and next pages, for spreads and dialogue that continues.",
         _obj({"range": {"type": "integer", "description": "1 or 2"}}, []), get_neighbors),
    Tool("local_ocr", "The local OCR model's reading of a region. A weak hint only: it is often wrong on handwriting.",
         _obj({"id": ID}, ["id"]), local_ocr),
    Tool("measure_bubble", "Size of a region's text box and roughly how many characters fit.",
         _obj({"id": ID}, ["id"]), measure_bubble),
    Tool("add_region", "Create a region for text that has no box. Give a rough box in page pixels, or grid cells "
         "of your last zoom; the program snaps it to the text strokes (strokes of other regions are left out) "
         "and shows what will be erased. Then set_text or skip_region the new id.",
         _obj({"box": BOX, "cells": {"type": "string", "description": "Grid cells of the last zoom, like C4 or B3:D5"},
               "direction": {"type": "string", "enum": ["v", "h"], "description": "v = vertical text, h = horizontal"},
               "after": {"type": "integer", "description": "Optional: put it after this region in reading order"},
               "exact": {"type": "boolean", "description": "Optional: take only the strokes whose middle lies in "
                         "the box, for a label close to other labels (by default nearby strokes of the same size "
                         "join, for text the box cut off)"},
               "reason": REASON}, ["direction", "reason"]), add_region, writes=True),
    Tool("merge_regions", "Merge regions that are one text block (for example one bubble split in two).",
         _obj({"ids": IDS, "reason": REASON}, ["ids", "reason"]), merge_regions, writes=True),
    Tool("split_region", "Split a region whose box holds two separate texts: a small aside or inner thought "
         "written beside the main text, a caption beside a title, a subtitle printed in a logo, a meaning gloss or a "
         "name written on the same detected line as invented glyphs. Give a box in page pixels, or grid cells of your "
         "last zoom: when it holds the middle of some of the region's lines, those whole lines move; otherwise the "
         "strokes inside it are cut out of the line (zoom first and box only the strokes of that part). Then "
         "set_text or skip_region both parts.",
         _obj({"id": ID, "box": BOX,
               "cells": {"type": "string", "description": "Grid cells of the last zoom, like C4 or B3:D5"},
               "direction": {"type": "string", "enum": ["v", "h"],
                             "description": "Optional: direction of the part that is cut out of a line, when it "
                                            "differs from the region's"},
               "reason": REASON}, ["id", "reason"]), split_region, writes=True),
    Tool("fit_region", "Put a region's box on its text strokes again: for a detected box that took in some of the "
         "art (its font size then reads too large), cut off part of the text, read lines written apart as one, or "
         "did not see that a label is written on a slant (a slanted line gets a slanted box). "
         "Give a box around the region's text in page pixels, or grid cells of your last zoom (default: its current "
         "box); strokes of other regions are left out. Its lines and font size follow the strokes, its text stays.",
         _obj({"id": ID, "box": BOX,
               "cells": {"type": "string", "description": "Grid cells of the last zoom, like C4 or B3:D5"},
               "direction": {"type": "string", "enum": ["v", "h"], "description": "Optional: v = vertical text, "
                             "h = horizontal, when the region's is wrong"},
               "reason": REASON}, ["id", "reason"]), fit_region, writes=True),
    Tool("skip_region", "Leave a region as it is in the original: not erased, not translated. Categories: "
         + ", ".join(SKIP_CATEGORIES) + ". Use not_text for detections that contain no text.",
         _obj({"id": ID, "category": {"type": "string", "enum": list(SKIP_CATEGORIES)}, "reason": REASON},
              ["id", "category", "reason"]), skip_region, writes=True),
    Tool("note_region", "Keep a region's text as drawn and set its translation beside it in brackets, smaller: "
         "for text written over detailed art (a map's labels on screentone between dashed lines, words across "
         "hatching, hair or an object), where erasing it would break the drawing; text on flat ground (paper, one "
         "flat color, its own box or circle, also when an outline touches it) is refused. Leave box out: the program sizes the note to be read "
         "on the page and puts it on the spot next to the text that covers the least art and no other text, and "
         "shows it; lines, screentone and panel borders under a note are fine. Only when that image shows the "
         "note covering a face or other text, call it again with box (page pixels or cells of your last zoom): "
         "an empty spot right beside the text to center it on (a spot further away is refused); its size stays. "
         "set_text the region to translate it in its place instead.",
         _obj({"id": ID,
               "source_text": {"type": "string", "description": "Exact transcription of the original text"},
               "translation": {"type": "string", "description": "The translation, without brackets (the program "
                               "adds them); [BR] marks a line break"},
               "category": {"type": "string", "enum": list(CATEGORIES)},
               "confidence": {"type": "number", "description": "0-1: how sure you are of reading and translation"},
               "box": {**BOX, "description": "Only to move a note already placed: [x1, y1, x2, y2] page pixels "
                                             "of the empty spot right beside the text to center it on"},
               "cells": {"type": "string", "description": "Only to move a note already placed: grid cells of the "
                                                          "last zoom to center it on, like C4 or B3:D5"},
               "direction": {"type": "string", "enum": ["v", "h"], "description": "Optional: v = vertical, "
                             "h = horizontal note (default: as the text)"},
               "reason": REASON},
              ["id", "source_text", "translation", "category", "confidence", "reason"]), note_region, writes=True),
    Tool("set_reading_order", "Reading order of the regions on this page (right-to-left manga order by default).",
         _obj({"ids": IDS}, ["ids"]), set_reading_order, writes=True),
    Tool("link_regions", "Mark regions as parts of one sentence that spans several bubbles.",
         _obj({"ids": IDS, "reason": REASON}, ["ids", "reason"]), link_regions, writes=True),
    Tool("set_text", "Save the exact source transcription and the translation of one region. "
         "Call it once per region; several calls in one turn are fine.",
         _obj({"id": ID,
               "source_text": {"type": "string", "description": "Exact transcription of the original text, without ruby"},
               "ruby": {"type": "string", "description": "Furigana / side readings, if any, as 'base(reading)' pairs"},
               "translation": {"type": "string", "description": "The translation that will be typeset; "
                               "[BR] marks a line break"},
               "category": {"type": "string", "enum": list(CATEGORIES)},
               "confidence": {"type": "number", "description": "0-1: how sure you are of reading and translation"},
               "style": {"type": "string", "description": "Only for lettering that clearly differs from the "
                         "normal dialogue: the name of its font style (set_font_style). Leave out for normal "
                         "dialogue and to keep the region's style; \"none\" goes back to the user's font."},
               "reason": REASON},
              ["id", "source_text", "translation", "category", "confidence"]), set_text, writes=True),
    Tool("set_font_style", "Define a font style: a kind of lettering that clearly differs from the normal "
         "dialogue (narration boxes, handwriting, a sign, a title ...) and the font on the font sheet closest to it. "
         "Then give that style in set_text for every region with that lettering.",
         _obj({"name": {"type": "string", "description": "Short name of the kind of lettering, like 旁白 or 手寫"},
               "font": {"type": "integer", "description": "Number of the closest font on the font sheet (1 or more)"},
               "reason": {"type": "string", "description": "What the original lettering looks like"}},
              ["name", "font", "reason"]), set_font_style, writes=True),
    Tool("set_line_breaks", "Put line breaks ([BR]) into a region's current translation without changing its words.",
         _obj({"id": ID, "text": {"type": "string", "description": "The current translation with [BR] where the "
                                  "lines break"}, "reason": REASON}, ["id", "text"]), set_line_breaks, writes=True),
    Tool("glossary_search", "Search the glossary (source or target).",
         _obj({"query": {"type": "string"}}, ["query"]), glossary_search),
    Tool("glossary_upsert", "Add or change a glossary term so later pages use the same translation.",
         _obj({"source": {"type": "string"}, "target": {"type": "string"}, "note": {"type": "string"},
               "category": {"type": "string", "enum": [*WORK_CATEGORIES, "Other"],
                            "description": "What the term is: a place, group, object, ability or creature name of "
                                           "this work (Location, Org, Item, Skill, Creature, Person), or Other for "
                                           "anything else (a sound, a one-off word). Names and terms may be added to "
                                           "the work's prompt file for later chapters; Other stays in this chapter."},
               "reason": REASON}, ["source", "target", "category", "reason"]), glossary_upsert, writes=True),
    Tool("character_get", "Look up characters (empty name lists all).",
         _obj({"name": {"type": "string"}}, []), character_get),
    Tool("character_upsert", "Add or update a character: translated name, reading, gender, how they speak.",
         _obj({"name": {"type": "string", "description": "Name as written in the source"},
               "target_name": {"type": "string"}, "reading": {"type": "string"}, "gender": {"type": "string"},
               "speech_style": {"type": "string", "description": "How they speak and address others"},
               "notes": {"type": "string", "description": "What they look like, and anything else that tells them apart"},
               "category": {"type": "string", "enum": ["Person", "Creature"],
                            "description": "Creature for an animal or monster character (default Person)"},
               "reason": REASON},
              ["name", "reason"]), character_upsert, writes=True),
    Tool("story_summary_get", "The running story summary.", _obj({}, []), story_summary_get),
    Tool("story_summary_update", "Replace the running story summary (keep it under ~300 words).",
         _obj({"summary": {"type": "string"}, "reason": REASON}, ["summary", "reason"]), story_summary_update,
         writes=True),
    Tool("search_past_translations", "Search earlier pages' source text and translations in this run.",
         _obj({"query": {"type": "string"}}, ["query"]), search_past_translations),
    Tool("flag_for_review", "Put something on the human review list and keep going. Types: " + ", ".join(FLAG_TYPES),
         _obj({"id": ID, "type": {"type": "string", "enum": list(FLAG_TYPES)}, "reason": REASON},
              ["type", "reason"]), flag_for_review, writes=True),
    Tool("mark_page_done", "Finish the page. Fails while any region has neither a translation nor a skip. The first "
         "call also returns the program checks (kana left, glossary, second reads, length) to fix; the second "
         "call finishes.",
         _obj({"confidence": {"type": "number", "description": "0-1 for the whole page"},
               "note": {"type": "string", "description": "Anything the human should know about this page"}},
              ["confidence"]), mark_page_done, writes=True),
    Tool("budget_status", "Context size and steps used on this page.", _obj({}, []), budget_status),
    Tool("undo", "Undo the last change of the regions on this page (flag_for_review entries stay).", _obj({}, []), undo,
         writes=True),
]


class ToolRegistry:
    def __init__(self, session: AgentSession, tools: Optional[List[Tool]] = None):
        self.session = session
        self.tools = {t.name: t for t in (tools or TOOLS)}

    def schemas(self) -> List[Dict[str, Any]]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}} for t in self.tools.values()]

    def prepare(self, name: str, raw_args: Any):
        """(tool, args, None), or (None, None, error result) when the call cannot run."""
        tool = self.tools.get(name)
        if tool is None:
            return None, None, ToolResult(
                text=f"Error: there is no tool named {name!r}. Available: {', '.join(self.tools)}.",
                ok=False, summary="unknown tool")
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            return None, None, ToolResult(text=f"Error: arguments are not valid JSON ({exc}).", ok=False,
                                          summary="bad JSON")
        if not isinstance(args, dict):
            return None, None, ToolResult(text="Error: arguments must be a JSON object.", ok=False,
                                          summary="bad arguments")
        missing = [p for p in tool.parameters.get("required", []) if args.get(p) in (None, "")]
        if missing:
            return None, None, ToolResult(text=f"Error: missing required argument(s): {', '.join(missing)}.",
                                          ok=False, summary="missing " + ", ".join(missing))
        if self.session.doc is None and name not in ("glossary_search", "character_get", "story_summary_get"):
            return None, None, ToolResult(text="Error: no page is open.", ok=False, summary="no page")
        return tool, args, None

    @staticmethod
    def failure(name: str, exc: Exception) -> ToolResult:
        if isinstance(exc, (RegionError, ValueError, KeyError, TypeError)):
            return ToolResult(text=f"Error: {exc}", ok=False, summary=str(exc)[:80])
        # a tool bug must not end the run
        logger.error(f"agent tool {name} failed", exc_info=exc)
        return ToolResult(text=f"Error: the tool failed internally ({type(exc).__name__}: {exc}).", ok=False,
                          summary=f"{type(exc).__name__}")

    def execute(self, name: str, raw_args: Any) -> ToolResult:
        tool, args, error = self.prepare(name, raw_args)
        if error is not None:
            return error
        try:
            return tool.handler(self.session, args)
        except Exception as exc:
            return self.failure(name, exc)
