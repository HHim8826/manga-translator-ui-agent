"""Checking again a chapter an earlier run finished ("revise").

The pages keep what they have: no preprocess, no new detection or translation. Each page is loaded
from its JSON (what the editor and the renderer use) and the agent's sidecar, and the translation
agent goes through it with every tool and the current rules: it splits asides and joined bubbles,
adds text that was missed (the recall check runs too), merges, skips or translates skipped text,
fixes translations and line breaks. A region it leaves alone is written back exactly as it was.

The user's own work stays:

* a region the user changed in the editor after the run (its text or its place), or drew there,
  counts as the user's (``Region.user_edited``). With ``revise_lock_user_edits`` (on by default) it
  is locked: the write tools refuse it, the proofreader and the font check leave it;
* a box the user deleted stays deleted (``PageDoc.removed``): neither the recall check nor
  add_region boxes its text again;
* a font the user gave a region stays (``Region.user_font``).

Thin fonts that earlier runs chose (MingLiU, Yu Mincho Light, SimSun, DFKai-SB ...) move to the
sheet font that looks most like them (``FontService.stand_in``); the font check then measures them
again. Locked regions keep theirs: the user laid their text out in it.

The inpainted image is the user's too (the editor's repairs are in it). Only what changed is
inpainted again (``inpaint_changes``): text that is erased now and was not before (a region added
or translated), and the original pixels come back where text is no longer erased (a region skipped
now). The render then reuses the image.

What a region was matched with: the sidecar's ``match`` box (``review.match_region``). Text is
compared loosely (``text_key``): the render writes vertical punctuation (︙ ︐ ︒ ││) and earlier runs
added line breaks after the agent, so those do not count as the user's changes.
"""
from __future__ import annotations

import base64
import copy
import logging
import os
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from .page import PageDoc, Region
from .review import match_region, region_bbox
from .store import AgentStore, _read_json

logger = logging.getLogger("manga_translator.agent")

MOVED_PX = 3
"""A region whose box moved more than this since the run was moved by the user."""
KEPT_SIDECAR = ("inpaint", "inpaint_score")
"""Sidecar fields of the earlier run that a check keeps: it does not review the inpainting again."""
_BREAKS = re.compile(r"\[BR\]|【BR】|<br\s*/?>", re.IGNORECASE)
_DOTS = re.compile(r"\.{2,}|…+")
_DASHES = re.compile(r"[—―─│┃︱ー〜~～-]+")


def text_key(text: Any) -> str:
    """``text`` without what the render and the line breaks change: line breaks, spaces, the forms of
    vertical punctuation (NFKC), runs of dots and of dashes."""
    text = _BREAKS.sub("", str(text or ""))
    text = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return _DASHES.sub("—", _DOTS.sub("…", text))


def _moved(box: Optional[Sequence[float]], other: Optional[Sequence[float]]) -> bool:
    if not box or not other or len(box) != 4 or len(other) != 4:
        return False
    return any(abs(float(a) - float(b)) > MOVED_PX for a, b in zip(box, other))


def _borrow_only(rich: Any, fallbacks: set) -> bool:
    """``translation_rich`` that only draws some characters from a fallback font (the font check's)."""
    def plain(node: Any) -> bool:
        if isinstance(node, dict):
            if node.get("type") not in (None, "text", "paragraph"):
                return False
            style = node.get("style") or {}
            if style and (set(style) - {"font_family"} or style.get("font_family") not in fallbacks):
                return False
            return all(plain(v) for k, v in node.items() if k != "style")
        if isinstance(node, list):
            return all(plain(v) for v in node)
        return True

    return bool(fallbacks) and isinstance(rich, dict) and plain(rich)


def _translation(data: dict) -> str:
    """The region's translation as the agent wrote it: ``translation_raw`` when it reads the same as
    ``translation`` (which has the render's vertical punctuation), else ``translation``."""
    shown = str(data.get("translation") or "")
    raw = str(data.get("translation_raw") or "")
    return raw if raw and text_key(raw) == text_key(shown) else shown


def _skipped_data(entry: dict) -> Optional[dict]:
    if isinstance(entry.get("json"), dict) and entry["json"].get("lines"):
        return copy.deepcopy(entry["json"])
    bbox = (entry.get("match") or {}).get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return {"lines": [quad], "texts": [], "text": "", "translation": "", "angle": 0, "font_size": 24,
            "direction": "v" if (y2 - y1) > (x2 - x1) else "h", "alignment": "center"}


def load_page(image_path: str, store: AgentStore, lock: bool = True, fonts=None, default_font: str = "",
              json_path: Optional[str] = None) -> PageDoc:
    """The page as the earlier run and the user left it, ready for the agent (see the module doc).
    ``fonts`` (``fonts.FontService``) replaces the thin fonts the earlier run chose; ``default_font``
    is the user's font (the render setting) when there is no font service."""
    doc = PageDoc.load(image_path, json_path)
    doc.revise = True
    side = store.load_sidecar(doc.stem) or {}
    entries = [e for e in side.get("regions") or [] if isinstance(e, dict)]
    font_info = side.get("fonts") or {}
    agent_fonts = {c.get("to") for c in font_info.get("changes") or [] if c.get("to")}
    fallbacks = {b.get("fallback") for b in font_info.get("borrowed") or [] if b.get("fallback")}
    default_font = getattr(fonts, "default_font", "") or default_font or ""
    stand_in: Optional[Callable[[str], Optional[str]]] = getattr(fonts, "stand_in", None)
    same: Callable[[str, str], bool] = getattr(fonts, "_same", None) or (lambda a, b: a.casefold() == b.casefold())

    def users_font(font: str, side_font: str) -> bool:
        """A font that neither the agent (its style, the font check) nor the user's settings gave."""
        if not font or not default_font:  # without the user's font nothing can be told apart
            return False
        return not any(same(font, f) for f in {side_font, default_font, *agent_fonts} if f)
    json_regions = [doc.regions[rid].data for rid in doc.order]

    # the agent's regions that are still there, and the ones the user deleted
    matched: Dict[int, dict] = {}
    entry_rid: Dict[Any, int] = {}
    removed: List[dict] = []
    for entry in entries:
        if entry.get("skipped") or entry.get("merged_into") is not None:
            continue
        match = entry.get("match") or {}
        index = match_region(match.get("center"), match.get("bbox"), json_regions)
        if index is None:
            if match.get("bbox"):
                removed.append({"bbox": [int(v) for v in match["bbox"]],
                                "text": str(entry.get("translation") or entry.get("ocr_agent")
                                            or entry.get("ocr_local") or "")[:60]})
            continue
        rid = doc.order[index]
        if rid not in matched:  # two agent regions in one box: the user joined them
            matched[rid] = entry
            entry_rid[entry.get("id")] = rid
    for item in side.get("removed") or []:  # deleted before an earlier check
        box = item.get("bbox") if isinstance(item, dict) else None
        if box and len(box) == 4 and match_region(None, [float(v) for v in box], json_regions) is None:
            removed.append({"bbox": [int(v) for v in box], "text": str(item.get("text") or "")})
    doc.removed = removed

    for rid in list(doc.order):
        region = doc.regions[rid]
        data = region.data
        entry = matched.get(rid)
        translation = _translation(data)
        region.loaded = copy.deepcopy(data)
        region.translation = translation or None
        source = str(data.get("text") or "".join(data.get("texts") or []) or "")
        region.source_text = source or None
        font = str(data.get("font_family") or "")
        if entry is None:  # drawn by the user
            region.user_edited = region.user_font = True
        else:
            region.ocr_local = str(entry.get("ocr_local") or region.ocr_local)
            region.category = str(entry.get("category") or "")
            region.ruby = str(entry.get("ruby") or "")
            confidence = entry.get("confidence")
            region.confidence = float(confidence) if isinstance(confidence, (int, float)) else None
            region.reasons = [str(r) for r in entry.get("reasons") or []]
            region.link_group = entry.get("link_group")
            region.origin = str(entry.get("origin") or "detector")
            region.style = str(entry.get("style") or "")
            side_font = str(entry.get("font") or "")
            region.user_edited = bool(entry.get("user_edited")) \
                or text_key(entry.get("translation")) != text_key(translation) \
                or _moved((entry.get("match") or {}).get("bbox"), region_bbox(data))
            region.user_font = bool(entry.get("user_font")) or users_font(font, side_font)
            if font and not region.user_font and default_font and not same(font, default_font):
                region.font = font  # the agent's font, kept in the sidecar
        region.locked = lock and region.user_edited
        # a locked region keeps its font too: the user laid its text out in that font (ch34 p6: a label
        # spaced with blank lines no longer met its frames in the stand-in's line height)
        if font and not region.user_font and not region.locked and stand_in is not None:
            new = stand_in(font)
            if new is not None:
                region.font = new or default_font
                region.drop_rich = _borrow_only(data.get("translation_rich"), fallbacks)
        region.loaded_state = (region.source_text, region.translation, data.get("lines"))
    if matched:
        doc._next_link = max([int(r.link_group) for r in doc.regions.values()
                              if isinstance(r.link_group, int)] + [0]) + 1

    # what the earlier run skipped is not in the JSON: back as skipped regions
    for entry in entries:
        if not entry.get("skipped") or entry.get("merged_into") is not None:
            continue
        match = entry.get("match") or {}
        if match.get("bbox") and match_region(match.get("center"), match.get("bbox"), json_regions) is not None:
            continue  # the user put it back in the editor
        data = _skipped_data(entry)
        if data is None:
            continue
        rid = max(doc.regions, default=0) + 1
        region = Region(id=rid, data=data, ocr_local=str(entry.get("ocr_local") or ""),
                        source_text=entry.get("ocr_agent") or None, category=str(entry.get("category") or ""),
                        skipped=True, skip_category=str(entry.get("skip_category") or "other"),
                        reasons=[str(r) for r in entry.get("reasons") or []],
                        origin=str(entry.get("origin") or "detector"), was_skipped=True)
        data["center"] = region.center
        doc.regions[rid] = region
        doc.order.append(rid)
        entry_rid[entry.get("id")] = rid
    _order_like(doc, side.get("order") or [e.get("id") for e in entries], entry_rid)

    doc.flags = [f for f in side.get("review_flags") or [] if isinstance(f, dict) and f.get("agent") == "inpaint"]
    doc.sidecar_extra = {k: side[k] for k in KEPT_SIDECAR if k in side}
    return doc


def _order_like(doc: PageDoc, side_order: Sequence[Any], entry_rid: Dict[Any, int]) -> None:
    """Skipped regions (added at the end) go back after the region they followed in the earlier run;
    the JSON keeps the order of the others."""
    restored = [rid for rid in doc.order if doc.regions[rid].was_skipped]
    order = [rid for rid in doc.order if rid not in restored]
    earlier = [entry_rid[e] for e in side_order if e in entry_rid]
    for rid in restored:
        if rid not in earlier:
            order.append(rid)
            continue
        place = earlier.index(rid)
        before = next((r for r in reversed(earlier[:place]) if r in order), None)
        order.insert(0 if before is None else order.index(before) + 1, rid)
    doc.order = order


def unskip_erase(doc: PageDoc, region: Region, page: np.ndarray) -> None:
    """Text the earlier run skipped is not in the saved mask: when the agent translates it now, its
    strokes are found again so the render erases them."""
    if not (doc.revise and region.was_skipped) or region.erase is not None or region.skipped:
        return
    from .snap import snap_text

    direction = "h" if region.direction.startswith("h") else "v"
    snap = snap_text(page, region.bbox, direction, exclude=doc.occupied(page.shape, but=region.id))
    if snap is not None:
        region.erase = (snap.erase_origin[0], snap.erase_origin[1], snap.erase)
        return
    # no strokes found (lettering the snap does not take): the boxes of its lines, so the old text
    # does not stay under the translation
    x1, y1, x2, y2 = region.bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(page.shape[1], x2), min(page.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    patch = np.zeros((y2 - y1, x2 - x1), np.uint8)
    polys = [np.round(p - (x1, y1)).astype(np.int32).reshape(-1, 1, 2) for p in region.polygons]
    cv2.fillPoly(patch, polys, 1)
    region.erase = (x1, y1, patch)


# --- the inpainted image ----------------------------------------------------------
def _mask(encoded: Any, shape) -> Optional[np.ndarray]:
    if not isinstance(encoded, str) or not encoded:
        return None
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape[:2] != tuple(shape[:2]):
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _page_mask(json_path: str):
    raw = _read_json(json_path, {}) if json_path and os.path.exists(json_path) else {}
    entry = raw.get(next(iter(raw)), {}) if isinstance(raw, dict) and raw else {}
    return entry.get("mask_raw") if isinstance(entry, dict) else None


INPAINT_GROW = 3
"""Pixels around newly erased strokes that are inpainted too (antialiased edges)."""


async def inpaint_changes(image_path: str, before_json: str, backend, save: Callable[[str, Any], Any]) -> Dict[str, Any]:
    """Bring the page's inpainted image up to its new mask, and leave the rest of it as it is.

    ``before_json`` is the page's JSON from before the check (the run's backup). Returns
    ``{"status": "unchanged" | "patched" | "no_inpainted" | "no_mask", "added": px, "restored": px}``;
    with ``no_inpainted`` the render inpaints the page itself."""
    from PIL import Image

    from ..utils.path_manager import find_inpainted_path, find_json_path

    inpainted_path = find_inpainted_path(image_path)
    if not inpainted_path or not os.path.exists(inpainted_path):
        return {"status": "no_inpainted", "added": 0, "restored": 0}
    old_encoded, new_encoded = _page_mask(before_json), _page_mask(find_json_path(image_path))
    if old_encoded == new_encoded:
        return {"status": "unchanged", "added": 0, "restored": 0}
    with Image.open(inpainted_path) as handle:
        inpainted = np.array(handle.convert("RGB"))
    old, new = _mask(old_encoded, inpainted.shape), _mask(new_encoded, inpainted.shape)
    if old is None or new is None:
        return {"status": "no_mask", "added": 0, "restored": 0}
    added, gone = new & ~old, old & ~new
    result = inpainted
    if gone.any():  # erased before, kept now: the original pixels
        with Image.open(image_path) as handle:
            source = np.array(handle.convert("RGB"))
        if source.shape != inpainted.shape:
            source = cv2.resize(source, (inpainted.shape[1], inpainted.shape[0]), interpolation=cv2.INTER_AREA)
        result = result.copy()
        result[gone] = source[gone]
    if added.any():
        grown = cv2.dilate(added.astype(np.uint8), np.ones((2 * INPAINT_GROW + 1,) * 2, np.uint8)) * 255
        method = (backend.methods() or [""])[0]
        if not method:
            raise RuntimeError("no inpainting model is available")
        result = np.asarray(await backend.inpaint(method, result, grown))
    save(image_path, result)
    return {"status": "patched", "added": int(added.sum()), "restored": int(gone.sum())}


def locked_centers(side: Optional[dict], key: str = "locked") -> set:
    """``match`` centers of the sidecar regions with ``key`` set (``locked``, ``user_font``), for the
    stages that read the JSON (the proofreader and the font check match regions by center)."""
    out = set()
    for entry in (side or {}).get("regions") or []:
        if isinstance(entry, dict) and entry.get(key) and not entry.get("skipped"):
            center = (entry.get("match") or {}).get("center")
            if center:
                out.add(tuple(float(v) for v in center))
    return out


def json_center(region: dict) -> tuple:
    """A JSON region's center the way the sidecar's ``match`` has it."""
    return tuple(float(v) for v in Region(id=0, data=region).center)
