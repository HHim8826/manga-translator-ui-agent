"""What the AI Agent left for the human on one page, for the editor (design doc, "編輯器").

The agents keep their notes in the page's sidecar (``agent/pages/<page>.agent.json``), not in the
JSON the editor loads. ``load`` turns it into three lists:

* ``flags``: the review list. Each agent numbers regions its own way: the translator by its own
  region ids (matched to the editor's boxes by position, ``match_region``), the proofreader and the
  font check by the JSON order, the inpaint review by its areas (a box on the page). Flags carry an
  ``agent`` field; older sidecars are sorted out by flag type.
* ``skipped``: regions the translator skipped (sound effects, signs...). They are not in the JSON;
  ``restore_data`` builds the editor region to add one back.
* ``units``: the inpaint review's areas with every attempt; ``use_attempt`` puts another attempt
  into the inpainted image.
"""
from __future__ import annotations

import copy
import io
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .store import AgentStore
from .store import page_stem

logger = logging.getLogger("manga_translator.agent")

Box = List[float]


@dataclass
class Flag:
    type: str
    reason: str
    agent: str
    """``translator``, ``proofread``, ``fonts`` or ``inpaint``."""
    number: Optional[int] = None
    """The flag's region number in its agent's own numbering; None for the whole page."""
    center: Optional[List[float]] = None
    bbox: Optional[Box] = None
    """Where on the page (translator regions and inpaint areas)."""
    json_index: Optional[int] = None
    """The region's place in the JSON (proofread and font flags)."""


@dataclass
class Skipped:
    id: int
    category: str
    text: str
    reason: str
    center: List[float]
    bbox: Box
    data: Optional[dict] = None
    """The region as it was in the JSON, kept in the sidecar by newer runs."""


@dataclass
class Attempt:
    n: int
    method: str
    verdict: List[str]
    file: str
    mask: str


@dataclass
class Unit:
    id: str
    box: List[int]
    chosen: int
    flagged: bool
    auto_fill: bool
    attempts: List[Attempt] = field(default_factory=list)


@dataclass
class PageReview:
    image_path: str
    sidecar_path: str
    status: str
    confidence: Optional[float]
    note: str
    flags: List[Flag]
    skipped: List[Skipped]
    units: List[Unit]


def _store(image_path: str) -> AgentStore:
    from ..utils.path_manager import get_work_dir

    return AgentStore(get_work_dir(image_path))


def _box(value) -> Optional[Box]:
    try:
        x1, y1, x2, y2 = (float(v) for v in value[:4])
    except (TypeError, ValueError, IndexError):
        return None
    return [x1, y1, x2, y2]


def _font_flag_numbers(side: dict) -> set:
    fonts = side.get("fonts") or {}
    numbers = {c.get("region") for c in fonts.get("changes") or [] if c.get("reason") == "layout"}
    return numbers | {u.get("region") for u in fonts.get("unfixed") or []}


def flag_agent(flag: dict, side: dict) -> str:
    """The agent a sidecar flag came from, flags from before they named it included (an old
    ``type=inpaint`` flag is the inpaint review's; an old ``layout`` flag on a font-check region the font
    check's). The review lists and a check that keeps the inpaint review's flags both go by it."""
    if flag.get("agent"):
        return str(flag["agent"])
    if flag.get("type") == "inpaint":  # before flags named their agent
        return "inpaint"
    if flag.get("type") == "layout" and flag.get("region") in _font_flag_numbers(side):
        return "fonts"
    return "translator"


def load(image_path: str) -> Optional[PageReview]:
    """The page's review lists; None when the AI Agent has not worked on it."""
    store = _store(image_path)
    stem = page_stem(image_path)
    side = store.load_sidecar(stem)
    if not isinstance(side, dict):
        return None
    regions = {r.get("id"): r for r in side.get("regions") or [] if isinstance(r, dict)}
    units = {}
    for uid, entry in ((side.get("inpaint") or {}).get("units") or {}).items():
        folder = store.candidates_dir(stem)
        attempts = [Attempt(n=i + 1, method=str(a.get("method") or ""), verdict=list(a.get("verdict") or []),
                            file=os.path.join(folder, a.get("file") or ""), mask=os.path.join(folder, a.get("mask") or ""))
                    for i, a in enumerate(entry.get("attempts") or [])]
        units[str(uid)] = Unit(id=str(uid), box=[int(v) for v in entry.get("box") or [0, 0, 0, 0]],
                               chosen=int(entry.get("chosen") or 1), flagged=bool(entry.get("flagged")),
                               auto_fill=bool(entry.get("auto_fill")), attempts=attempts)
    flags = []
    for raw in side.get("review_flags") or []:
        if not isinstance(raw, dict):
            continue
        agent = flag_agent(raw, side)
        number = raw.get("region")
        number = int(number) if isinstance(number, (int, float)) or str(number).isdigit() else None
        flag = Flag(type=str(raw.get("type") or ""), reason=str(raw.get("reason") or ""), agent=agent, number=number)
        if number is not None and agent == "translator" and number in regions:
            match = regions[number].get("match") or {}
            flag.center, flag.bbox = match.get("center"), _box(match.get("bbox"))
        elif number is not None and agent == "inpaint" and str(number) in units:
            x1, y1, x2, y2 = units[str(number)].box
            flag.bbox, flag.center = [x1, y1, x2, y2], [(x1 + x2) / 2, (y1 + y2) / 2]
        elif number is not None and agent in ("proofread", "fonts"):
            box = _box(raw.get("bbox"))
            if box is not None:  # where the region was: found again after the user reorders the boxes
                center = raw.get("center")
                flag.bbox = box
                flag.center = list(center) if isinstance(center, (list, tuple)) and len(center) == 2 else \
                    [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            else:  # flags from before they kept the position: the JSON order at the time
                flag.json_index = number - 1
        flags.append(flag)
    skipped = []
    for region in side.get("regions") or []:
        if not isinstance(region, dict) or not region.get("skipped") or region.get("merged_into"):
            continue
        match = region.get("match") or {}
        bbox = _box(match.get("bbox"))
        if bbox is None:
            continue
        reasons = region.get("reasons") or []
        skipped.append(Skipped(id=int(region.get("id") or 0), category=str(region.get("skip_category") or "other"),
                               text=str(region.get("ocr_agent") or region.get("ocr_local") or ""),
                               reason=str(reasons[-1]) if reasons else "",
                               center=list(match.get("center") or [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]),
                               bbox=bbox, data=region.get("json") if isinstance(region.get("json"), dict) else None))
    confidence = side.get("confidence")
    return PageReview(image_path=image_path, sidecar_path=store.sidecar_path(stem), status=str(side.get("status") or ""),
                      confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                      note=str(side.get("note") or ""), flags=flags, skipped=skipped, units=list(units.values()))


# --- the editor's regions -----------------------------------------------------------
def region_bbox(region: dict) -> Optional[Box]:
    """The box around an editor (JSON) region's lines."""
    points = [p for polygon in (region.get("lines") or region.get("polygons") or []) for p in polygon or []]
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return None
    return [min(xs), min(ys), max(xs), max(ys)] if xs else None


def _overlap(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_region(center: Optional[Sequence[float]], bbox: Optional[Box], regions: Sequence[dict]) -> Optional[int]:
    """The editor region at the agent's box: the most overlapping one, or the one whose box holds
    the agent's center. None when no region is there (deleted or moved in the editor)."""
    if bbox is None:
        return None
    return _best_box(center, bbox, [region_bbox(region) for region in regions])


def _best_box(center, bbox: Box, boxes: Sequence[Optional[Box]]) -> Optional[int]:
    best, best_score = None, 0.0
    for index, box in enumerate(boxes):
        if box is None:
            continue
        score = _overlap(bbox, box)
        if center is not None and box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]:
            score = max(score, 0.3)
        if score > best_score:
            best, best_score = index, score
    return best if best_score >= 0.3 else None


def _plain(values, n: int) -> bool:
    try:
        return len(values) >= n and all(isinstance(v, (int, float)) and math.isfinite(v) for v in values[:n])
    except TypeError:
        return False


class RegionBoxes:
    """``match_region`` for many boxes against the same regions (a check matches every region of the
    earlier run): each region's box is worked out once and all of them are scored together, with the
    same scores, the same 0.3 and the first of equal boxes. Anything but plain numbers goes the way
    ``match_region`` goes. 250 regions against 250 built 62,500 boxes before."""

    def __init__(self, regions: Sequence[dict]):
        self.boxes = [region_bbox(region) for region in regions]
        self.valid = np.array([b is not None for b in self.boxes], dtype=bool)
        self.array = np.array([b if b is not None else [0.0] * 4 for b in self.boxes],
                              dtype=np.float64).reshape(-1, 4)
        self.plain = bool(np.isfinite(self.array).all())

    def match(self, center: Optional[Sequence[float]], bbox: Optional[Box]) -> Optional[int]:
        if bbox is None:
            return None
        if not self.plain or not _plain(bbox, 4) or (center is not None and not _plain(center, 2)):
            return _best_box(center, bbox, self.boxes)
        if not self.boxes:
            return None
        a0, a1, a2, a3 = (float(v) for v in bbox[:4])
        b0, b1, b2, b3 = self.array.T
        w = np.minimum(a2, b2) - np.maximum(a0, b0)
        h = np.minimum(a3, b3) - np.maximum(a1, b1)
        inter = w * h
        union = (a2 - a0) * (a3 - a1) + (b2 - b0) * (b3 - b1) - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where((w > 0) & (h > 0) & (union > 0), inter / union, 0.0)
        if center is not None:
            c0, c1 = float(center[0]), float(center[1])
            inside = (b0 <= c0) & (c0 <= b2) & (b1 <= c1) & (c1 <= b3)
            score = np.where(inside, np.maximum(score, 0.3), score)
        score = np.where(self.valid, score, 0.0)
        best = int(np.argmax(score))  # the first of equal scores, like the loop
        return best if score[best] >= 0.3 else None


def flag_region(flag: Flag, regions: Sequence[dict]) -> Optional[int]:
    """The editor region a flag is about, if any: the box at the flag's position (None when it was
    deleted or moved away), or for old proofreading and font flags the JSON order."""
    if flag.agent != "inpaint" and flag.bbox is not None:
        return match_region(flag.center, flag.bbox, regions)
    if flag.json_index is not None:
        return flag.json_index if 0 <= flag.json_index < len(regions) else None
    return None


STYLE_KEYS = ("font_family", "font_size", "font_color", "bg_colors", "alignment", "line_spacing", "letter_spacing",
              "stroke_width", "stroke_color")
NEW_BOX = {"font_family": "", "font_color": "#000000", "bg_colors": [255, 255, 255], "alignment": "auto",
           "line_spacing": 1.0, "letter_spacing": 1.0, "stroke_width": 0.07}


def restore_data(item: Skipped, regions: Sequence[dict]) -> dict:
    """An editor region for a skipped one: as it was in the JSON when the sidecar kept it, otherwise
    a box at its place with its text, styled like the nearest region of the page."""
    if item.data:
        data = copy.deepcopy(item.data)
    else:
        x1, y1, x2, y2 = item.bbox
        quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        data = {"lines": [quad], "polygons": [copy.deepcopy(quad)], "font_size": 24, "angle": 0,
                "direction": "v" if (y2 - y1) > (x2 - x1) else "h"}
        nearest = None
        for region in regions:
            box = region_bbox(region)
            if box is None:
                continue
            d = ((box[0] + box[2]) / 2 - item.center[0]) ** 2 + ((box[1] + box[3]) / 2 - item.center[1]) ** 2
            if nearest is None or d < nearest[0]:
                nearest = (d, region)
        if nearest is not None:
            data.update({k: copy.deepcopy(nearest[1][k]) for k in STYLE_KEYS if k in nearest[1]})
        for key, value in NEW_BOX.items():  # what the editor gives a box drawn by hand
            data.setdefault(key, copy.deepcopy(value))
    data.update({"text": item.text, "texts": [item.text], "translation": "", "translation_raw": "",
                 "center": list(item.center)})
    data.pop("translation_rich", None)
    return data


# --- inpaint attempts -----------------------------------------------------------------
def _read_rgb(path: str) -> np.ndarray:
    from PIL import Image

    with open(path, "rb") as fh:
        image = Image.open(io.BytesIO(fh.read()))
        image.load()
    return np.array(image.convert("RGB"))


def _read_gray(path: str) -> np.ndarray:
    from PIL import Image

    with open(path, "rb") as fh:
        image = Image.open(io.BytesIO(fh.read()))
        image.load()
    return np.array(image.convert("L"))


def use_attempt(review: PageReview, unit_id: str, n: int) -> str:
    """Put attempt ``n`` of an inpaint area into the page's inpainted image (and the JSON mask, and
    the sidecar's choice); returns the inpainted image's path."""
    from PIL import Image

    from ..utils.path_manager import find_inpainted_path, find_json_path
    from .inpaint import decode_mask, encode_mask

    unit = next((u for u in review.units if u.id == str(unit_id)), None)
    if unit is None or not 1 <= n <= len(unit.attempts):
        raise ValueError(f"no attempt {n} for area {unit_id}")
    target = find_inpainted_path(review.image_path)
    if not target or not os.path.exists(target):
        raise FileNotFoundError("the page has no inpainted image")
    old, new = unit.attempts[unit.chosen - 1] if 1 <= unit.chosen <= len(unit.attempts) else None, unit.attempts[n - 1]
    x1, y1, x2, y2 = unit.box
    original = _read_rgb(review.image_path)
    image = _read_rgb(target)
    if image.shape != original.shape:
        raise ValueError("the inpainted image does not match the page")
    patch = _read_rgb(new.file)
    new_mask = _read_gray(new.mask) > 0
    old_mask = _read_gray(old.mask) > 0 if old is not None else np.zeros_like(new_mask)
    if patch.shape[:2] != (y2 - y1, x2 - x1) or new_mask.shape != patch.shape[:2] or old_mask.shape != new_mask.shape:
        raise ValueError("the attempt does not fit its area")
    view = image[y1:y2, x1:x2]
    view[old_mask] = original[y1:y2, x1:x2][old_mask]  # what the chosen attempt replaced goes back
    view[new_mask] = patch[new_mask]

    # everything is worked out first; then the image, the JSON mask and the sidecar's choice are
    # replaced together, and put back as they were when one of them cannot be written
    json_path = find_json_path(review.image_path)
    data = None
    if json_path and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data.values():
            if isinstance(entry, dict) and isinstance(entry.get("mask_raw"), str):
                mask = decode_mask(entry["mask_raw"])
                if mask is not None and mask.shape[:2] == original.shape[:2]:
                    full = mask > 0
                    full[y1:y2, x1:x2] = (full[y1:y2, x1:x2] & ~old_mask) | new_mask
                    entry["mask_raw"] = encode_mask(full)
    with open(review.sidecar_path, "r", encoding="utf-8") as fh:
        side = json.load(fh)
    saved = ((side.get("inpaint") or {}).get("units") or {}).get(str(unit_id))
    if isinstance(saved, dict):
        saved["chosen"] = n
        saved["picked_in_editor"] = True

    staged: List[tuple] = []  # (temporary file, target)
    try:
        staged.append((_stage_image(target, image), target))
        if data is not None:
            staged.append((_stage_json(json_path, data), json_path))
        staged.append((_stage_json(review.sidecar_path, side), review.sidecar_path))
        _commit(staged)
    finally:
        for tmp, _ in staged:
            if os.path.exists(tmp):
                os.remove(tmp)
    unit.chosen = n
    return target


def _stage_image(target: str, image: np.ndarray) -> str:
    from PIL import Image

    folder, name = os.path.split(target)
    tmp = os.path.join(folder, f".tmp_{os.getpid()}_{name}")  # same extension: same image format
    Image.fromarray(image).save(tmp)
    return tmp


def _stage_json(target: str, data: Any) -> str:
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(target))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return tmp


def _commit(staged: Sequence[tuple]) -> None:
    """Move each staged file over its target. When one move fails, the targets already replaced get
    their old content back and the error is raised."""
    old = []
    for _, target in staged:
        with open(target, "rb") as fh:
            old.append(fh.read())
    done = 0
    try:
        for tmp, target in staged:
            os.replace(tmp, target)
            done += 1
    except BaseException:
        for (_, target), content in zip(staged[:done], old[:done]):
            try:
                with open(target, "wb") as fh:
                    fh.write(content)
            except OSError:
                logger.exception(f"could not put {target} back")
        raise
