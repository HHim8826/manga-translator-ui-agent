"""The inpainting review agent.

After the translation agent finished a chapter, every finished page goes
through one short, separate conversation: the program inpaints the page block
by block at native size and runs the checks from ``quality_checks``; the
model then sees, for every area, the original with the mask and the result
side by side, with the program's findings marked, and fixes what is wrong
with the precise tools (numbers, grid cells, parameters). Nothing of this
enters the translation conversation.

At the end the chosen attempts are put together into the page's inpainted
image (render reuses it), the JSON mask is updated, every attempt is kept as a
candidate, and the inpaint score lowers the page confidence.
"""
from __future__ import annotations

import copy
import inspect
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from . import images
from .events import EventLog
from .inpaint import FILL, Attempt, InpaintBackend, InpaintError, InpaintPage, Unit, decode_mask, encode_mask
from . import json_guard
from .page import PageDoc, Region
from .settings import AgentSettings
from .store import AgentStore, write_json_atomic
from .store import page_stem
from .tools import CELL, REASON, Tool, ToolRegistry, ToolResult, _obj

logger = logging.getLogger("manga_translator.agent")

MASK_TINT = (255, 40, 40)
OTHER_TINT = (120, 120, 120)
KEEP_TINT = (0, 170, 90)
"""Text kept as drawn on purpose (``InpaintPage.keep``)."""
RESIDUAL_COLOR = (230, 0, 0)
BROKEN_COLOR = (255, 140, 0)
STROKE_COLOR = (0, 110, 255)

REVIEW_PROMPT = """You review the text removal (inpainting) of one manga page inside a manga translation app. Translated text will be typeset over these areas later, so every area must look as if the erased text had never been there, and the art around it must stay intact.

You are not a coding assistant: you act only by calling the tools provided.

The first message shows the page with numbered areas, then a sheet with one pair per area: left = the original with this area's mask (the pixels that get replaced) tinted red and other areas' masks gray, right = the result. The program checked every result: red boxes = text the detector still finds there (residual text, reliable), orange boxes = art lines that run into the mask and may not continue (only a hint: check by eye).

Text kept as drawn: the translator left some text on the page on purpose (sound effects, invented script and its kana readings, logos and their subtitles, credits, signs). It is boxed "keep" on the overview and tinted green on the sheets and views. It is not residual text: never try to add it to a mask (the tools leave it out anyway), and do not flag it as text left over.

Look at every area. Problems and their fixes, cheapest first:
- Residual text (ghost strokes, outlines, a missed small character, furigana): adjust_mask dilate 2-4 px with scope residual; or view_region show=components and mask_select the stroke numbers to add. Then inpaint_region.
- Damaged art (a bubble outline or panel border cut or smudged, hair or lines erased): protect_lines; or mask_select remove for mask parts that cover no text; or mask_paint remove. Then inpaint_region.
- Unnatural fill (blur, smears, streaks, broken screentone, a line not continued): inpaint_region with another method; fill gives a clean flat color in plain white or single-color bubbles.
- A flat fill where the background is not really flat (a gradient, screentone, art): pick_candidate 1 (the model's result) or inpaint_region with a model.
Areas on a flat background were already filled with its color in the first pass, keeping lines that cross the mask as drawn; the model's result for them is attempt 1.
Every inpaint_region starts again from the original, with the current mask. After {retries} new attempts on one area, pick_candidate the best attempt and flag_for_review it.
flag_for_review is for problems you could not fix (residual text, damaged art, a visible smear or a broken line): the human has to repair those. When an area is fine but worth a glance (text erased on a face, across a panel border, or over a large area of screentone or detailed art), write that with note_area instead: a note stays with the page and does not go on the review list.
Leave good areas alone. When every area is fine, noted or flagged, call finish_review.

Write every reason and flag in {reason_lang}; a human reads them."""

HUMAN = {
    "CHT": {"residual": "修圖後仍偵測到殘留文字", "steps": "修圖審查在 {steps} 步後停止，沒有完成",
            "low": "修圖與程式檢查的信心分數 {score:.2f}", "error": "修圖審查失敗：{error}"},
    "CHS": {"residual": "修图后仍检测到残留文字", "steps": "修图审查在 {steps} 步后停止，没有完成",
            "low": "修图与程序检查的置信分数 {score:.2f}", "error": "修图审查失败：{error}"},
    "ENG": {"residual": "text is still detected after inpainting", "steps": "the inpaint review stopped after {steps} steps",
            "low": "inpaint and program check score {score:.2f}", "error": "inpaint review failed: {error}"},
}


def _human(lang: str, key: str, **fields) -> str:
    return (HUMAN.get(lang) or HUMAN["ENG"])[key].format(**fields)


def with_kept(doc: PageDoc, side: Optional[dict]) -> List[Region]:
    """Put the text the translator kept as drawn back into ``doc`` from the sidecar (a page loaded
    from its JSON has no skipped regions: they are left out of it), so ``doc.keep_mask`` sees it, and
    mark the notes set beside such text (``note_region``, found by their place): nothing is erased
    under them. Returns the kept regions."""
    entries = [e for e in (side or {}).get("regions", []) if isinstance(e, dict)]
    notes = [((e.get("match") or {}).get("center"), e["note_of"]) for e in entries
             if e.get("note_of") is not None and not e.get("skipped")]
    for region in doc.regions.values():
        cx, cy = region.center
        for center, of in notes:
            if isinstance(center, (list, tuple)) and len(center) == 2 and \
                    abs(cx - float(center[0])) <= 2 and abs(cy - float(center[1])) <= 2:
                region.note_of = int(of)  # the translator's id of the text it belongs to
    next_id = max(doc.regions, default=0) + 1
    for entry in entries:
        data = entry.get("json")
        if not entry.get("skipped") or entry.get("merged_into") is not None or not isinstance(data, dict) \
                or not data.get("lines"):
            continue
        doc.regions[next_id] = Region(id=next_id, data=copy.deepcopy(data), skipped=True,
                                      skip_category=str(entry.get("skip_category") or "other"))
        doc.order.append(next_id)
        next_id += 1
    return doc.kept_drawn()


# --- images --------------------------------------------------------------------------
def _tint(crop: np.ndarray, mask: np.ndarray, color=MASK_TINT, alpha: float = 0.45) -> np.ndarray:
    out = crop.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def _scale_for(unit: Unit, side: int, max_scale: float = 4.0) -> float:
    w, h = unit.size
    return max(0.25, min(max_scale, side / max(1, max(w, h))))


def _resize(array: np.ndarray, scale: float) -> Image.Image:
    image = Image.fromarray(array)
    return image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.LANCZOS)


def _boxes(draw: ImageDraw.ImageDraw, boxes: Sequence[Sequence[int]], origin: Tuple[int, int], scale: float,
           color, offset_x: int = 0, offset_y: int = 0) -> None:
    ox, oy = origin
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((offset_x + (x1 - ox) * scale - 2, offset_y + (y1 - oy) * scale - 2,
                        offset_x + (x2 - ox) * scale + 2, offset_y + (y2 - oy) * scale + 2), outline=color, width=3)


def _grid(draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int) -> None:
    columns, rows = images.GRID_COLUMNS, images.GRID_ROWS
    font = images._font(max(11, min(18, w // 30)))
    cw, ch = w / len(columns), h / rows
    for i in range(1, len(columns)):
        draw.line([(x0 + i * cw, y0), (x0 + i * cw, y0 + h)], fill=(0, 140, 255), width=1)
    for j in range(1, rows):
        draw.line([(x0, y0 + j * ch), (x0 + w, y0 + j * ch)], fill=(0, 140, 255), width=1)
    for i, col in enumerate(columns):
        for j in range(rows):
            draw.text((x0 + i * cw + 2, y0 + j * ch + 1), f"{col}{j + 1}", fill=(0, 110, 230), font=font)


def pair_image(page: InpaintPage, unit: Unit, composite: np.ndarray, side: int = 300, grid: bool = False,
               attempt: Optional[Attempt] = None, mask: Optional[np.ndarray] = None) -> Image.Image:
    """Original with the mask tinted | result, findings boxed, with a label strip on top."""
    attempt = attempt or unit.current
    scale = _scale_for(unit, side)
    shown_mask = attempt.mask if mask is None else mask
    others = page.others_mask(unit) & ~shown_mask  # other areas' masks in gray: not this area's business
    left = _tint(_tint(page.crop(unit), others, OTHER_TINT, alpha=0.35), shown_mask)
    left = _resize(_tint(left, page.kept(unit), KEEP_TINT, alpha=0.3), scale)
    result_crop = page.crop(unit, composite).copy()
    if attempt is not unit.current:  # show the attempt itself, not the chosen one
        result_crop[attempt.mask] = attempt.result[attempt.mask]
    right = _resize(result_crop, scale)
    gap, strip = 8, 26
    out = Image.new("RGB", (left.width + right.width + gap, left.height + strip), "white")
    out.paste(left, (0, strip))
    out.paste(right, (left.width + gap, strip))
    draw = ImageDraw.Draw(out)
    origin = unit.box[:2]
    _boxes(draw, attempt.residual, origin, scale, RESIDUAL_COLOR, left.width + gap, strip)
    _boxes(draw, attempt.broken, origin, scale, BROKEN_COLOR, left.width + gap, strip)
    if grid:
        _grid(draw, 0, strip, left.width, left.height)
    font = images._font(18)
    findings = []
    if attempt.residual:
        findings.append(f"residual {len(attempt.residual)}")
    if attempt.broken:
        findings.append(f"lines {len(attempt.broken)}")
    label = f"#{unit.id}  a{attempt.n}/{len(unit.attempts)} {attempt.method}" + \
            (" · " + " · ".join(findings) if findings else "")
    draw.text((4, 3), label, fill=RESIDUAL_COLOR if attempt.residual else (0, 0, 0), font=font)
    return out


def mask_preview(page: InpaintPage, unit: Unit, side: int = 520) -> Image.Image:
    """The working mask after an edit, tinted on the original."""
    scale = _scale_for(unit, side)
    return _resize(_tint(_tint(page.crop(unit), unit.work), page.kept(unit), KEEP_TINT, alpha=0.3), scale)


def components_image(page: InpaintPage, unit: Unit, mask_comps: List[Dict], strokes: List[Dict],
                     side: int = 640) -> Image.Image:
    scale = _scale_for(unit, side, max_scale=6.0)
    crop = _tint(_tint(page.crop(unit), unit.work, alpha=0.3), page.kept(unit), KEEP_TINT, alpha=0.3)
    for comp in strokes:
        crop = _tint(crop, comp["pixels"], STROKE_COLOR, alpha=0.6)
    out = _resize(crop, scale)
    draw = ImageDraw.Draw(out)
    font = images._font(max(14, min(22, out.width // 25)))
    for comp, color in [(c, RESIDUAL_COLOR) for c in mask_comps] + [(c, STROKE_COLOR) for c in strokes]:
        x1, y1, x2, y2 = comp["box"]
        draw.rectangle((x1 * scale - 1, y1 * scale - 1, x2 * scale + 1, y2 * scale + 1), outline=color, width=2)
        tw = draw.textlength(comp["name"], font=font)
        ty = max(0, y1 * scale - font.size - 4)
        draw.rectangle((x1 * scale, ty, x1 * scale + tw + 6, ty + font.size + 4), fill=color)
        draw.text((x1 * scale + 3, ty + 1), comp["name"], fill="white", font=font)
    return out


def contact_sheets(page: InpaintPage, composite: np.ndarray, max_width: int = 1500,
                   max_height: int = 1500) -> List[Image.Image]:
    pairs = [pair_image(page, unit, composite) for unit in page.units]
    sheets: List[Image.Image] = []
    rows: List[List[Image.Image]] = [[]]
    width = 0
    for pair in pairs:
        if rows[-1] and width + pair.width + 16 > max_width:
            rows.append([])
            width = 0
        rows[-1].append(pair)
        width += pair.width + 16

    def flush(block: List[List[Image.Image]]) -> None:
        if not block:
            return
        w = max(sum(p.width + 16 for p in row) for row in block)
        h = sum(max(p.height for p in row) + 16 for row in block)
        sheet = Image.new("RGB", (w, h), (235, 235, 235))
        y = 0
        for row in block:
            x = 0
            for p in row:
                sheet.paste(p, (x, y))
                x += p.width + 16
            y += max(p.height for p in row) + 16
        sheets.append(sheet)

    block: List[List[Image.Image]] = []
    height = 0
    for row in rows:
        row_h = max(p.height for p in row) + 16
        if block and height + row_h > max_height:
            flush(block)
            block, height = [], 0
        block.append(row)
        height += row_h
    flush(block)
    return sheets


# --- session and tools ---------------------------------------------------------------
class ReviewSession:
    def __init__(self, settings: AgentSettings, page: InpaintPage, page_image: Image.Image, methods: List[str],
                 target_lang: str = "", events: Optional[EventLog] = None, stem: str = ""):
        self.settings = settings
        self.doc = page  # ToolRegistry checks that a page is open
        self.page = page
        self.page_image = page_image
        self.methods = methods
        self.target_lang = target_lang
        self.events = events
        self.stem = stem
        self.flags: List[Dict[str, Any]] = []
        self.finished = False
        self.finish_warned = False
        self.note = ""
        self.steps = 0

    @property
    def lang(self) -> str:
        return self.target_lang if self.target_lang in HUMAN else "ENG"

    def unit(self, a: Dict[str, Any]) -> Unit:
        unit = self.page.unit(a.get("id"))
        unit.steps += 1
        if unit.steps > self.settings.ops_per_region:
            raise InpaintError(f"#{unit.id} used its {self.settings.ops_per_region} operations: pick_candidate the "
                               "best attempt and flag_for_review it")
        return unit

    def flag(self, unit: Optional[Unit], reason: str, source: str = "agent") -> Dict[str, Any]:
        entry = {"region": unit.id if unit else None, "type": "inpaint", "reason": reason, "agent": "inpaint"}
        if unit:
            unit.flagged = True
        self.flags.append(entry)
        if self.events:
            self.events.emit("flag", page=self.stem, agent="inpaint", region=entry["region"], flag="inpaint",
                             reason=reason, source=source)
        return entry


def _url(image: Image.Image) -> str:
    return images.data_url(image, max_side=1600)


def view_region(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    unit = s.unit(a)
    show = a.get("show") or "compare"
    if show == "components":
        mask_comps, strokes = s.page.components(unit)
        image = components_image(s.page, unit, mask_comps, strokes)
        text = (f"#{unit.id}: red M1-M{len(mask_comps)} = parts of the current mask; "
                + (f"blue S1-S{len(strokes)} = strokes near the mask that it does not cover"
                   if strokes else "no uncovered strokes near the mask")
                + ". Use these numbers with mask_select, or M numbers with adjust_mask scope components.")
        return ToolResult(text=text, images=[_url(image)], summary=f"#{unit.id} M{len(mask_comps)} S{len(strokes)}")
    if show != "compare":
        raise InpaintError("show must be compare or components")
    composite = s.page.composite()
    image = pair_image(s.page, unit, composite, side=640, grid=True)
    w, h = unit.size
    unit.view = {"w": w, "h": h}
    attempts = "\n".join(("* " if i == unit.chosen else "  ") + at.describe() for i, at in enumerate(unit.attempts))
    pending = f"\nMask edits not inpainted yet: {', '.join(unit.work_ops)}." if unit.history else ""
    return ToolResult(text=(f"#{unit.id} (left: original + mask with the A1-H8 grid for mask_paint, right: result):\n"
                            f"{attempts}{pending}"),
                      images=[_url(image)], summary=f"#{unit.id} a{unit.current.n}")


def _edited(s: ReviewSession, unit: Unit, label: str, before: int, after: int, extra: str = "") -> ToolResult:
    image = mask_preview(s.page, unit)
    if s.page.kept_out:
        extra += (f"; {s.page.kept_out} px of it were text kept as drawn (green), which stays out of the mask: "
                  "leave that text alone")
    return ToolResult(text=(f"#{unit.id}: {label}; mask {before} → {after} px{extra}. Not applied yet: call "
                            "inpaint_region to inpaint with this mask, or undo."),
                      images=[_url(image)], summary=f"#{unit.id} {label}")


def mask_select(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    unit = s.unit(a)
    label, before, after = s.page.select(unit, a.get("add") or [], a.get("remove") or [])
    return _edited(s, unit, label, before, after)


def adjust_mask(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    unit = s.unit(a)
    label, before, after = s.page.adjust(unit, str(a.get("op", "")), int(a.get("px", 0)),
                                         str(a.get("scope") or "all"), a.get("components") or [])
    return _edited(s, unit, label, before, after)


def mask_paint(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    unit = s.unit(a)
    if not unit.view:
        raise InpaintError(f"cells refer to the grid of view_region #{unit.id}; call it first")
    match = CELL.fullmatch(str(a.get("cells", "")))
    if not match:
        raise InpaintError("cells must look like C4 or B3:D5 (grid of the last view_region)")
    columns = images.GRID_COLUMNS
    c1, r1 = columns.index(match[1].upper()), int(match[2]) - 1
    c2, r2 = (columns.index(match[3].upper()), int(match[4]) - 1) if match[3] else (c1, r1)
    c1, c2 = sorted((c1, c2))
    r1, r2 = sorted((r1, r2))
    cw, ch = unit.view["w"] / len(columns), unit.view["h"] / images.GRID_ROWS
    box = (c1 * cw, r1 * ch, (c2 + 1) * cw, (r2 + 1) * ch)
    label, before, after = s.page.paint(unit, box, str(a.get("mode", "")), bool(a.get("ink_only", True)))
    return _edited(s, unit, f"{label} {a.get('cells')}", before, after)


def protect_lines(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    unit = s.unit(a)
    label, before, after, lines = s.page.protect_lines(unit, int(a.get("width") or 2))
    return _edited(s, unit, label, before, after, extra=f" ({lines} line(s) kept out)")


async def inpaint_region(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    unit = s.unit(a)
    method = str(a.get("method") or s.methods[0])
    if method not in s.methods:
        raise InpaintError(f"method must be one of: {', '.join(s.methods)}")
    if unit.retries >= s.settings.inpaint_attempts_per_region:
        raise InpaintError(f"#{unit.id} already has {unit.retries} new attempts: pick_candidate the best one "
                           "and flag_for_review it")
    previous = unit.current
    attempt = await s.page.retry(unit, method)
    image = pair_image(s.page, unit, s.page.composite(), side=520)
    text = (f"#{unit.id}: {attempt.describe()} (before: residual {len(previous.residual)}, broken lines "
            f"{len(previous.broken)}). It is now the chosen attempt; pick_candidate to go back.")
    return ToolResult(text=text, images=[_url(image)],
                      summary=f"#{unit.id} a{attempt.n} {method}: " + (", ".join(attempt.issues) or "ok"))


def pick_candidate(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    # not one of the region's operations: the limit tells the model to pick one, and counted, the pick
    # hit the same limit (review 2026-11-28 B9)
    unit = s.page.unit(a.get("id"))
    attempt = s.page.pick(unit, int(a.get("n", 0)))
    image = pair_image(s.page, unit, s.page.composite(), side=400)
    return ToolResult(text=f"#{unit.id}: chose {attempt.describe()}.", images=[_url(image)],
                      summary=f"#{unit.id} a{attempt.n}")


def undo(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    unit = s.page.unit(a.get("id"))  # not counted either: it takes an operation back
    label = s.page.undo(unit)
    return ToolResult(text=f"#{unit.id}: undid {label}; mask {int(unit.work.sum())} px.", summary=f"#{unit.id} {label}")


def flag_for_review(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    unit = s.page.unit(a.get("id")) if a.get("id") not in (None, "") else None
    s.flag(unit, str(a.get("reason", "")))
    return ToolResult(text="Added to the review list; the chosen attempt is kept. Keep going.",
                      summary=(f"#{unit.id} " if unit else "") + "inpaint")


def note_area(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    unit = s.page.unit(a.get("id"))
    note = str(a.get("note") or "").strip()
    if not note:
        raise InpaintError("note is required")
    unit.note = note
    return ToolResult(text="Noted with the page (not on the review list). Keep going.", summary=f"#{unit.id} note")


def finish_review(s: ReviewSession, a: Dict[str, Any]) -> ToolResult:
    left = [u for u in s.page.units if u.current.residual and not u.flagged]
    if left and not s.finish_warned:
        s.finish_warned = True
        ids = ", ".join(f"#{u.id}" for u in left)
        return ToolResult(text=(f"Not finished yet: residual text is still detected in {ids}. Fix them, or call "
                                "finish_review again to keep them as they are (they go to the review list)."),
                          ok=False, summary=f"residual {ids}")
    s.finished = True
    s.note = str(a.get("note") or "")
    return ToolResult(text="Review finished.", summary="done")


def review_tools(methods: List[str]) -> List[Tool]:
    uid = {"type": "integer", "description": "Area number as drawn on the page and the sheet"}
    numbers = {"type": "array", "items": {"type": "integer"}}
    return [
        Tool("view_region", "Larger view of one area. show=compare (default): original + mask | result with an "
             "A1-H8 grid and all attempts listed. show=components: the mask split into numbered parts (M1..) and "
             "the strokes near it that it does not cover (S1..).",
             _obj({"id": uid, "show": {"type": "string", "enum": ["compare", "components"]}}, ["id"]), view_region),
        Tool("mask_select", "Add strokes (S numbers) to the mask and/or remove mask parts (M numbers) of an area, "
             "from its last components view.",
             _obj({"id": uid, "add": {**numbers, "description": "S numbers to add"},
                   "remove": {**numbers, "description": "M numbers to remove"}, "reason": REASON},
                  ["id", "reason"]), mask_select, writes=True),
        Tool("adjust_mask", "Grow (dilate) or shrink (erode) the mask by px pixels: everywhere (scope all), only "
             "around the residual text boxes (scope residual), or only around some mask parts (scope components "
             "with M numbers).",
             _obj({"id": uid, "op": {"type": "string", "enum": ["dilate", "erode"]},
                   "px": {"type": "integer", "description": "1-12"},
                   "scope": {"type": "string", "enum": ["all", "residual", "components"]},
                   "components": {**numbers, "description": "M numbers for scope components"}, "reason": REASON},
                  ["id", "op", "px", "reason"]), adjust_mask, writes=True),
        Tool("mask_paint", "Add or remove the mask in grid cells of the last view_region of this area. add takes "
             "only the ink in those cells unless ink_only is false.",
             _obj({"id": uid, "cells": {"type": "string", "description": "Like C4 or B3:D5"},
                   "mode": {"type": "string", "enum": ["add", "remove"]},
                   "ink_only": {"type": "boolean"}, "reason": REASON}, ["id", "cells", "mode", "reason"]),
             mask_paint, writes=True),
        Tool("protect_lines", "Take art lines that cross the mask (bubble outlines, panel borders, hair) out of it, "
             "with a band of width px around them, so they stay as drawn.",
             _obj({"id": uid, "width": {"type": "integer", "description": "1-6, default 2"}, "reason": REASON},
                  ["id", "reason"]), protect_lines, writes=True),
        Tool("inpaint_region", "Inpaint the area again from the original with its current mask. Methods: "
             + ", ".join(methods) + (" (fill = flat color sampled around the mask)" if FILL in methods else "")
             + ". The program checks the result.",
             _obj({"id": uid, "method": {"type": "string", "enum": methods}, "reason": REASON}, ["id", "reason"]),
             inpaint_region, writes=True),
        Tool("pick_candidate", "Use attempt n of an area (numbers from view_region).",
             _obj({"id": uid, "n": {"type": "integer"}, "reason": REASON}, ["id", "n", "reason"]), pick_candidate,
             writes=True),
        Tool("undo", "Undo the last mask edit of an area that was not inpainted yet.", _obj({"id": uid}, ["id"]),
             undo, writes=True),
        Tool("flag_for_review", "Put an area (or the whole page, without id) on the human review list for a "
             "problem you could not fix; its chosen attempt is kept, and the human can switch to another attempt "
             "in the editor.",
             _obj({"id": uid, "reason": REASON}, ["reason"]), flag_for_review, writes=True),
        Tool("note_area", "A note on an area that is fine but worth a glance (erased on a face, across a panel "
             "border, over large screentone): kept with the page, not put on the review list.",
             _obj({"id": uid, "note": {"type": "string"}}, ["id", "note"]), note_area, writes=True),
        Tool("finish_review", "Finish the review of this page.",
             _obj({"note": {"type": "string", "description": "Anything the human should know"}}, []),
             finish_review, writes=True),
    ]


class AsyncToolRegistry(ToolRegistry):
    async def aexecute(self, name: str, raw_args: Any) -> ToolResult:
        tool, args, error = self.prepare(name, raw_args)
        if error is not None:
            return error
        try:
            result = tool.handler(self.session, args)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            return self.failure(name, exc)


# --- the conversation ---------------------------------------------------------------
class InpaintReviewer:
    def __init__(self, settings: AgentSettings, client, store: AgentStore, events: EventLog, target_lang: str,
                 backend: InpaintBackend, should_stop: Callable[[], bool] = lambda: False,
                 save_image: Optional[Callable[[str, np.ndarray], Any]] = None):
        self.settings = settings
        self.client = client
        self.store = store
        self.events = events
        self.target_lang = target_lang
        self.backend = backend
        self.should_stop = should_stop
        self.save_image = save_image
        self.totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "requests": 0}

    def system_prompt(self) -> str:
        from .runner import language_name
        return REVIEW_PROMPT.format(retries=self.settings.inpaint_attempts_per_region,
                                    reason_lang=language_name(self.target_lang))

    def _first_message(self, session: ReviewSession) -> dict:
        page = session.page
        composite = page.composite()
        marks = [("keep", list(box), "skipped") for box in page.kept_boxes]
        marks += [(u.id, list(u.box), "pending" if u.current.issues else "done") for u in page.units]
        overview = images.draw_marks(Image.fromarray(composite), marks)
        lines = []
        for unit in page.units:
            attempt = unit.current
            what = "mask area without a region" if unit.extra else \
                "region " + ", ".join(f"#{r}" for r in unit.regions)
            findings = []
            if attempt.residual:
                findings.append(f"residual text at {len(attempt.residual)} spot(s) (red)")
            if attempt.broken:
                findings.append(f"{len(attempt.broken)} of {attempt.entries} line(s) may be broken (orange)")
            how = "flat fill" if unit.auto_fill else attempt.method
            lines.append(f"#{unit.id} ({what}, {unit.size[0]}x{unit.size[1]} px, {how}): "
                         + ("; ".join(findings) if findings else "no program findings"))
        filled = sum(1 for u in page.units if u.auto_fill)
        method = page.first_method or page.units[0].attempts[0].method
        first = f"method {method}" + (
            f"; {filled} area(s) on a flat background were filled with its color instead (attempt 2, the "
            f"{method} result is attempt 1)" if filled else "")
        kept = (f"\n{len(page.kept_boxes)} box(es) of text kept as drawn (gray \"keep\" on the overview, green "
                f"on the sheets): leave that text as it is." if page.kept_boxes else "")
        text = (f"Page {session.stem}: {len(page.units)} inpainted area(s), {first}. "
                f"Methods for inpaint_region: {', '.join(session.methods)}.\n" + "\n".join(lines) + kept
                + "\nThe overview (green = no findings, red = findings) comes first, then the sheet(s).")
        content: List[dict] = [{"type": "text", "text": text},
                               {"type": "image_url", "image_url": {"url": images.data_url(overview, max_side=1280),
                                                                   "detail": self.settings.image_detail}}]
        for sheet in contact_sheets(page, composite):
            content.append({"type": "image_url", "image_url": {"url": images.data_url(sheet), "detail": "high"}})
        return {"role": "user", "content": content}

    async def review_page(self, image_path: str) -> Dict[str, Any]:
        from .runner import AgentStopped, _compact_args

        doc = PageDoc.load(image_path)
        stem = doc.stem
        if json_guard.taken(doc.json_path):  # saved in the editor during the run: the user's
            return {"page": stem, "image": image_path, "status": "skipped", "reason": "saved in the editor"}
        encoded = doc.page_data.get("mask_raw")
        mask = decode_mask(encoded) if isinstance(encoded, str) else None
        methods = list(self.backend.methods())
        if mask is None or not methods:
            # a page with no mask has nothing to review; one without an inpainter waits for a redo
            return {"page": stem, "image": image_path, "status": "skipped",
                    "reason": "no mask" if mask is None else "no inpainter", "nothing_to_do": mask is None}
        kept = with_kept(doc, self.store.load_sidecar(stem))
        # a note has nothing of its own to erase under it
        regions = [(r.id, r.polygons, tuple(r.bbox)) for r in doc.active_regions()
                   if not r.skipped and r.note_of is None]
        page_image = images.open_page(image_path)
        array = np.asarray(page_image)
        page = InpaintPage(array, mask, regions, self.backend, stem, keep=doc.keep_mask(array.shape),
                           kept_boxes=[tuple(r.bbox) for r in kept])
        if not page.units:
            return {"page": stem, "image": image_path, "status": "skipped", "reason": "empty mask",
                    "nothing_to_do": True}
        if FILL not in methods:
            methods.append(FILL)
        await page.initial(methods[0], self.settings.fill_flat_areas)
        start = dict(self.totals)
        session = ReviewSession(self.settings, page, page_image, methods, self.target_lang, self.events, stem)
        registry = AsyncToolRegistry(session, review_tools(methods))
        found = sum(1 for u in page.units if u.current.issues)
        self.events.emit("inpaint_start", page=stem, agent="inpaint", units=len(page.units), issues=found,
                         method=methods[0], filled=sum(1 for u in page.units if u.auto_fill))
        messages: List[dict] = [{"role": "system", "content": self.system_prompt()}, self._first_message(session)]
        for step in range(self.settings.inpaint_review_steps):
            if self.should_stop():
                raise AgentStopped()
            session.steps = step + 1
            reply = await self.client.complete(messages, registry.schemas(), "required",
                                               effort=self.settings.review_reasoning_effort,
                                               cache_key=f"inpaint-{stem}")
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                self.totals[key] += getattr(reply, key)
            self.totals["requests"] += 1
            assistant: Dict[str, Any] = {"role": "assistant", "content": reply.content or None}
            if reply.tool_calls:
                assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                            "function": {"name": c.name, "arguments": c.arguments}}
                                           for c in reply.tool_calls]
            messages.append(assistant)
            if not reply.tool_calls:
                messages.append({"role": "user", "content": "Continue with tool calls; finish with finish_review."})
                continue
            for call in reply.tool_calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": call.arguments}
                self.events.emit("tool_call", page=stem, agent="inpaint", tool=call.name, args=_compact_args(args),
                                 reason=args.get("reason", "") if isinstance(args, dict) else "")
                result = await registry.aexecute(call.name, call.arguments)
                self.events.emit("tool_result", page=stem, agent="inpaint", tool=call.name, ok=result.ok,
                                 summary=result.summary, images=len(result.images))
                content: Any = result.text
                if result.images:
                    content = [{"type": "text", "text": result.text}] + [
                        {"type": "image_url", "image_url": {"url": url, "detail": "high"}} for url in result.images]
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            if session.finished:
                break
        lang = session.lang
        if not session.finished:
            session.flag(None, _human(lang, "steps", steps=session.steps), source="check")
        for unit in page.units:
            if unit.current.residual and not unit.flagged:
                session.flag(unit, _human(lang, "residual"), source="check")
        usage = {key: self.totals[key] - start[key] for key in self.totals}
        return self._save(doc, page, session, usage)

    def _save(self, doc: PageDoc, page: InpaintPage, session: ReviewSession,
              usage: Dict[str, int]) -> Dict[str, Any]:
        stem = doc.stem
        if json_guard.taken(doc.json_path):  # saved in the editor during the review: the user's
            self.events.emit("inpaint_done", page=stem, agent="inpaint", status="done", skipped="edited",
                             units=len(page.units), retried=0, filled=0, flagged=0, image=doc.image_path)
            return {"page": stem, "image": doc.image_path, "status": "skipped", "reason": "saved in the editor"}
        composite = page.composite()
        saved = None
        if self.save_image is not None:
            # raises when the image was not written: the render must not reuse an older one
            saved = self.save_image(doc.image_path, composite)
        with open(doc.json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        entry = data[doc.key]
        entry["mask_raw"] = encode_mask(page.final_mask())
        entry["mask_is_refined"] = True
        write_json_atomic(doc.json_path, data)
        json_guard.note(doc.json_path)

        candidates = page.save_candidates(self.store.candidates_dir(stem))
        side = self.store.load_sidecar(stem) or {}
        side.pop("version", None)
        flags = list(side.get("review_flags") or []) + session.flags
        score = page.score()
        before = side.get("confidence")
        confidence = round(min(score, before), 3) if isinstance(before, (int, float)) else score
        if confidence < self.settings.low_confidence_threshold and not flags:
            low = {"region": None, "type": "low_confidence", "reason": _human(session.lang, "low", score=confidence),
                   "agent": "inpaint"}
            flags.append(low)
            self.events.emit("flag", page=stem, agent="inpaint", region=None, flag="low_confidence",
                             reason=low["reason"], source="check")
        side.update({"review_flags": flags, "confidence": confidence, "inpaint_score": score,
                     "inpaint": {"units": candidates, "note": session.note, "steps": session.steps,
                                 "finished": session.finished, "usage": usage}})
        self.store.save_sidecar(stem, side)
        retried = sum(1 for u in page.units if u.retries)
        filled = sum(1 for u in page.units if u.auto_fill)
        flagged = sum(1 for u in page.units if u.flagged) + sum(1 for f in session.flags if f["region"] is None)
        inpainted = None
        if self.save_image is not None:  # where the Agents page preview finds the result
            from ..utils.path_manager import find_inpainted_path
            inpainted = saved if isinstance(saved, str) and saved else find_inpainted_path(doc.image_path)
        self.events.emit("inpaint_done", page=stem, agent="inpaint", status=side.get("status") or "done",
                         units=len(page.units), retried=retried, filled=filled, flagged=flagged, score=score,
                         confidence=confidence, flags=len(flags), steps=session.steps, usage=usage,
                         image=doc.image_path, inpainted=inpainted)
        return {"page": stem, "image": doc.image_path, "status": "reviewed", "units": len(page.units),
                "retried": retried, "filled": filled, "flagged": flagged, "score": score, "confidence": confidence}

    async def run(self, image_paths: Sequence[str]) -> List[Dict[str, Any]]:
        from .runner import AgentStopped
        outcomes = []
        for path in image_paths:
            if self.should_stop():
                raise AgentStopped()
            stem = page_stem(path)
            try:
                outcomes.append(await self.review_page(path))
            except AgentStopped:
                raise
            except Exception as exc:
                from .client import quota_resets

                logger.exception(f"inpaint review of {stem} failed")
                resets = quota_resets(exc)  # a usage limit: when to do it again by hand (redo.py)
                self.events.emit("error", page=stem, agent="inpaint", error=f"{type(exc).__name__}: {str(exc)[:300]}",
                                 redo=True, **resets)
                outcomes.append({"page": stem, "image": path, "status": "error", "error": str(exc), **resets})
        return outcomes
