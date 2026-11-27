"""Inpainting of one page for the review agent.

The page mask (the refined mask in the page JSON, after the translation agent's
edits) is split into *units*: the mask components of one kept region. A unit
has a box around it, a working mask and a list of attempts; each attempt is
the inpainted pixels under its mask. The page result is the original with
every unit's chosen attempt pasted in under that attempt's mask, so pixels
outside the mask never change and a retry always starts again from the
original (only the other units' results are kept as context).

Precise operations work on numbers the program
draws, not on coordinates: mask components M1..Mn and stroke candidates
S1..Sn, grid cells of the last view, and parameters such as "dilate 3 px
around the residual text". The checks from ``inpainting/quality_checks.py``
run after every attempt.

The inpainting models are reached through an ``InpaintBackend`` so tests can
run without them.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from ..inpainting import quality_checks as qc
from .snap import MAX_ELONGATION

Box = Tuple[int, int, int, int]
FILL = "fill"
FLAT_SPREAD = 12.0
FLAT_SPECKS = 0.02
SPECK_RING = 10
SPECK_AREA = 30
FLAT_ART = 0.03
ART_REACH = 4
"""90th percentile of the color distance around a mask above which a flat fill shows."""


def _write_png(path: str, image: np.ndarray) -> None:
    """``image`` as a PNG at ``path``, also outside the system code page (``cv2.imwrite`` returns
    False there and writes nothing); OSError when it cannot be written."""
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"could not encode {path}")
    with open(path, "wb") as fh:
        fh.write(data.tobytes())


class InpaintBackend(Protocol):
    def methods(self) -> List[str]:
        """Inpainting models that can run without a download, configured one first."""

    async def inpaint(self, method: str, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint ``image`` (RGB) under ``mask`` (uint8, >0 = fill), block by block at native size."""

    async def text_map(self, image: np.ndarray) -> Optional[np.ndarray]:
        """The text detector's text map (0-255) of ``image``, for the residual text check."""


@dataclass
class Attempt:
    n: int
    method: str
    mask: np.ndarray
    """Bool patch in the unit box: the pixels this attempt replaced."""
    result: np.ndarray
    """RGB patch in the unit box."""
    ops: List[str]
    note: str = ""
    residual: List[Box] = field(default_factory=list)
    broken: List[Box] = field(default_factory=list)
    entries: int = 0
    checked: bool = False

    @property
    def issues(self) -> List[str]:
        out = []
        if self.residual:
            out.append("residual_text")
        if self.broken:
            out.append("broken_lines")
        return out

    def describe(self) -> str:
        how = self.method + (f" after {', '.join(self.ops)}" if self.ops else "")
        checks = (f"residual text {len(self.residual)}, broken lines {len(self.broken)}/{self.entries}"
                  if self.checked else "not checked")
        return f"attempt {self.n} ({how}): {checks}" + (f"; {self.note}" if self.note else "")


@dataclass
class Unit:
    id: int
    regions: List[int]
    box: Box
    base_mask: np.ndarray
    """The page mask inside the box that belongs to this unit (bool)."""
    attempts: List[Attempt] = field(default_factory=list)
    chosen: int = 0
    work: Optional[np.ndarray] = None
    """Mask for the next attempt; mask edits change it."""
    work_ops: List[str] = field(default_factory=list)
    history: List[Tuple[np.ndarray, List[str]]] = field(default_factory=list)
    steps: int = 0
    retries: int = 0
    flagged: bool = False
    extra: bool = False
    """Mask area that belongs to no kept region."""
    auto_fill: bool = False
    """The first pass used the flat fill (attempt 2) for this unit."""
    view: Optional[Dict] = None
    """Grid of the last view of this unit, for mask_paint."""
    note: str = ""
    """What a human may want to look at in an area the reviewer found acceptable (note_area): kept
    with the page, not put on the review list."""

    @property
    def current(self) -> Attempt:
        return self.attempts[self.chosen]

    @property
    def size(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return x2 - x1, y2 - y1


class InpaintError(ValueError):
    """A request the model can correct (bad number, nothing to do...)."""


# --- helpers -----------------------------------------------------------------------
def _kernel(radius: int) -> np.ndarray:
    size = 2 * max(1, int(radius)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def decode_mask(encoded: str) -> Optional[np.ndarray]:
    mask = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    return mask[:, :, 0] if mask.ndim == 3 else mask


def encode_mask(mask: np.ndarray) -> str:
    ok, png = cv2.imencode(".png", np.where(mask > 0, 255, 0).astype(np.uint8))
    if not ok:
        raise RuntimeError("could not encode the mask")
    return base64.b64encode(png.tobytes()).decode("ascii")


@dataclass
class Background:
    color: np.ndarray
    spread: float
    """99th percentile of how far the paper under and around the mask strays from ``color`` (max channel)."""
    specks: float
    """Share of a wider ring around the mask covered by small ink specks (screentone dots)."""
    art: float = 0.0
    """Share of the mask covered by ink of shapes that reach well outside it (art, not text)."""
    keep: Optional[np.ndarray] = None
    """Those shapes inside the mask, 1px wider (bool): a flat fill leaves them as drawn."""
    ends: int = 0
    """End points of those lines at the mask: lines the text interrupts, which a fill cannot join."""

    @property
    def plain(self) -> bool:
        return self.spread <= FLAT_SPREAD and self.specks <= FLAT_SPECKS

    @property
    def flat(self) -> bool:
        return self.plain and self.art <= FLAT_ART and self.ends == 0


def background(crop: np.ndarray, mask: np.ndarray) -> Background:
    """Paper color around the mask and how flat it is. Only the paper connected to the mask counts
    (inside a closed bubble or box, the outline ends it). Ink and a 2px halo around it are left out,
    so text strokes and the anti-aliased edges of outlines do not count; screentone shows up as
    many small ink specks around the mask instead."""
    m = mask.astype(np.uint8)
    inner = cv2.dilate(m, _kernel(2)) > 0
    inky = ink(crop, mask)
    blank = ~inky
    _n, labels = cv2.connectedComponents(blank.astype(np.uint8), connectivity=4)
    touching = np.unique(labels[(mask if (mask & blank).any() else inner) & blank])
    region = np.isin(labels, touching[touching > 0]) & blank
    reach = cv2.dilate(region.astype(np.uint8), _kernel(3)) > 0  # with the specks inside it
    ring = (cv2.dilate(m, _kernel(5)) > 0) & ~inner & reach
    halo = cv2.dilate(inky.astype(np.uint8), _kernel(2)) > 0
    area = (mask | ring) & ~halo  # the paper under the text and right around it
    spread = 255.0
    if area.sum() >= 10:
        sample = area
    else:  # nearly all ink: take what there is, but it does not count as flat
        sample = next((s for s in (ring & ~inky, mask & ~inky, ring, ~mask) if s.any()), mask)
    pixels = crop[sample]
    if len(pixels) == 0:
        raise InpaintError("no background around the mask to take a color from")
    color = np.median(pixels.reshape(-1, crop.shape[2]), axis=0)
    if area.sum() >= 10:
        spread = float(np.percentile(np.abs(pixels.astype(np.float32) - color).max(axis=1), 99))
    wide = (cv2.dilate(m, _kernel(SPECK_RING)) > 0) & ~inner & reach
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inky.astype(np.uint8), connectivity=8)
    small = np.zeros(count, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < SPECK_AREA
    specks = float((small[labels] & wide).sum()) / max(1, int(wide.sum()))
    # ink under the mask that belongs to shapes reaching well outside it: art lines, hair, dense tone
    outside = labels[inky & ~(cv2.dilate(m, _kernel(ART_REACH)) > 0)]
    crossing = np.zeros(count, bool)
    crossing[np.unique(outside)] = True
    crossing[0] = False
    shapes = crossing[labels]
    art = float((shapes & mask).sum()) / max(1, int(mask.sum()))
    keep = (cv2.dilate(shapes.astype(np.uint8), _kernel(1)) > 0) & mask
    # a line that ends at the text (the text interrupts it) needs a model to join it again
    touching = np.zeros(count, bool)
    touching[np.unique(labels[inky & inner])] = True
    lines = (crossing & touching)[labels] & (cv2.dilate(m, _kernel(12)) > 0)
    ends = int((_line_ends(lines) & inner).sum()) if lines.any() else 0
    return Background(color, spread, specks, art, keep, ends)


def _line_ends(binary: np.ndarray) -> np.ndarray:
    """End points of the one-pixel skeleton of ``binary`` (bool)."""
    image = binary.astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc"):
        skeleton = cv2.ximgproc.thinning(image) > 0
    else:  # pragma: no cover - builds without opencv-contrib
        from skimage.morphology import skeletonize
        skeleton = skeletonize(binary)
    neighbours = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.uint8),
                              borderType=cv2.BORDER_CONSTANT)
    return skeleton & (neighbours == 2)


def fill_patch(crop: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, str]:
    """Flat fill with the color around the mask; the note warns when that may show."""
    bg = background(crop, mask)
    out = crop.copy()
    out[mask] = bg.color.astype(np.uint8)
    if not bg.plain:
        note = f"the background around it is not flat (spread {bg.spread:.0f}), so a flat fill may show"
    elif not bg.flat:
        note = "art lines run into the mask and the fill paints over them; protect_lines keeps them"
    else:
        note = ""
    return out, note


def flat_fill(crop: np.ndarray, mask: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """First-pass fill for an area on a flat background: (patch, the mask it replaced, note), or None.
    Art shapes that cross the mask (bubble outlines, lines) stay as drawn."""
    bg = background(crop, mask)
    if not bg.flat:
        return None
    filled = mask & ~bg.keep
    if not filled.any():
        return None
    out = crop.copy()
    out[filled] = bg.color.astype(np.uint8)
    note = "flat background: filled with its color" + \
        (", lines crossing the mask kept as drawn" if bg.keep.any() else "")
    return out, filled, note


def ink(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ink pixels of ``crop`` (bool): Otsu on the area around the mask; the
    strokes are the minority under the mask (dark text, or light text on a dark background)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    near = cv2.dilate(mask.astype(np.uint8), _kernel(6)) > 0
    sample = gray[near] if near.any() else gray.ravel()
    if sample.size < 2 or float(sample.std()) < 4.0:
        return np.zeros(gray.shape, bool)
    threshold, _ = cv2.threshold(sample.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = gray <= threshold  # Otsu returns the lower value itself on a two-tone image
    inside = dark[mask] if mask.any() else dark[near]
    return ~dark if inside.size and inside.mean() > 0.5 else dark


def _components(binary: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Labels and per-component info, numbered top to bottom, then right to left."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    dist = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    thickness = np.zeros(count, np.float32)
    np.maximum.at(thickness, labels.ravel(), dist.ravel())
    comps = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        comps.append({"label": label, "box": (x, y, x + w, y + h), "area": area,
                      "textlike": max(w, h) <= MAX_ELONGATION * max(1.0, 2.0 * float(thickness[label]))})
    comps.sort(key=lambda c: (c["box"][1] // 12, -c["box"][2]))
    return labels, comps


# --- page --------------------------------------------------------------------------
class InpaintPage:
    """One page: original, units, attempts, and the precise mask operations."""

    def __init__(self, image: np.ndarray, mask: np.ndarray, regions: Sequence[Tuple[int, List[np.ndarray], Box]],
                 backend: InpaintBackend, stem: str = "", keep: Optional[np.ndarray] = None,
                 kept_boxes: Sequence[Box] = ()):
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.original = np.ascontiguousarray(image[:, :, :3])
        self.keep = np.zeros(image.shape[:2], bool)
        """Text kept as drawn on purpose (``PageDoc.keep_mask``): no mask or mask edit takes it."""
        if keep is not None:
            if keep.shape[:2] != image.shape[:2]:
                keep = cv2.resize(keep.astype(np.uint8), (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            self.keep = keep > 0
        self.kept_boxes = [tuple(int(v) for v in box) for box in kept_boxes]
        """Boxes of that text, drawn on the overview and the sheets."""
        self.kept_out = 0
        """Pixels of kept text the last mask edit reached and left out."""
        self.mask = (mask > 0) & ~self.keep
        self.backend = backend
        self.stem = stem
        self.first_method = ""
        self.units: List[Unit] = []
        self.by_id: Dict[int, Unit] = {}
        self._build_units(regions)

    # --- units ---------------------------------------------------------------------
    def _build_units(self, regions: Sequence[Tuple[int, List[np.ndarray], Box]]) -> None:
        height, width = self.mask.shape
        count, labels, stats, _ = cv2.connectedComponentsWithStats(self.mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            return
        overlap = np.zeros((len(regions), count), np.int64)
        for index, (_rid, polygons, _box) in enumerate(regions):
            canvas = np.zeros((height, width), np.uint8)
            polys = [np.round(np.asarray(p)).astype(np.int32).reshape(-1, 1, 2) for p in polygons]
            if polys:
                cv2.fillPoly(canvas, polys, 1)
            canvas = cv2.dilate(canvas, _kernel(6))
            overlap[index] = np.bincount(labels[canvas > 0], minlength=count)[:count]
        owner: Dict[int, int] = {}
        for label in range(1, count):
            column = overlap[:, label] if len(regions) else np.zeros(0)
            if column.size and column.max() > 0:
                owner[label] = int(column.argmax())
                continue
            x, y, w, h, _ = (int(v) for v in stats[label])
            best, best_gap = None, 41
            for index, (_rid, _polys, rbox) in enumerate(regions):
                gap = max(0, max(x, rbox[0]) - min(x + w, rbox[2]), max(y, rbox[1]) - min(y + h, rbox[3]))
                if gap < best_gap:
                    best, best_gap = index, gap
            owner[label] = best if best is not None else -label
        groups: Dict[int, List[int]] = {}
        for label, index in owner.items():
            groups.setdefault(index, []).append(label)
        max_id = max((rid for rid, _p, _b in regions), default=0)
        next_extra = (max_id // 100 + 1) * 100 + 1
        for index, members in sorted(groups.items(), key=lambda kv: (kv[0] < 0, abs(kv[0]))):
            xs1 = [int(stats[m][0]) for m in members]
            ys1 = [int(stats[m][1]) for m in members]
            xs2 = [int(stats[m][0] + stats[m][2]) for m in members]
            ys2 = [int(stats[m][1] + stats[m][3]) for m in members]
            if index >= 0:
                rid, _polys, rbox = regions[index]
                xs1.append(int(rbox[0])); ys1.append(int(rbox[1])); xs2.append(int(rbox[2])); ys2.append(int(rbox[3]))
                touched = [regions[i][0] for i in range(len(regions)) if overlap[i, members].sum() > 0]
                uid, extra = rid, False
            else:
                uid, extra, touched = next_extra, True, []
                next_extra += 1
            x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
            pad = int(max(24, min(80, 0.3 * max(x2 - x1, y2 - y1))))
            box = (max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad))
            bx1, by1, bx2, by2 = box
            base = np.isin(labels[by1:by2, bx1:bx2], members)
            unit = Unit(id=uid, regions=touched or ([uid] if not extra else []), box=box, base_mask=base,
                        work=base.copy(), extra=extra)
            self.units.append(unit)
            self.by_id[uid] = unit

    def unit(self, uid) -> Unit:
        try:
            key = int(str(uid).lstrip("#"))
        except (TypeError, ValueError):
            raise InpaintError(f"{uid!r} is not a region number")
        if key not in self.by_id:
            known = ", ".join(f"#{u.id}" for u in self.units) or "none"
            raise InpaintError(f"#{key} has no inpainted area on this page (areas: {known})")
        return self.by_id[key]

    def crop(self, unit: Unit, image: Optional[np.ndarray] = None) -> np.ndarray:
        x1, y1, x2, y2 = unit.box
        return (self.original if image is None else image)[y1:y2, x1:x2]

    def kept(self, unit: Unit) -> np.ndarray:
        """The text kept as drawn inside this unit's box (bool)."""
        x1, y1, x2, y2 = unit.box
        return self.keep[y1:y2, x1:x2]

    def others_mask(self, unit: Unit) -> np.ndarray:
        """Other units' current masks inside this unit's box (bool)."""
        x1, y1, x2, y2 = unit.box
        out = np.zeros((y2 - y1, x2 - x1), bool)
        for other in self.units:
            if other is unit:
                continue
            ox1, oy1, ox2, oy2 = other.box
            ix1, iy1, ix2, iy2 = max(x1, ox1), max(y1, oy1), min(x2, ox2), min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            mask = other.current.mask if other.attempts else other.base_mask
            out[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] |= mask[iy1 - oy1:iy2 - oy1, ix1 - ox1:ix2 - ox1]
        return out

    # --- attempts ------------------------------------------------------------------
    def composite(self, skip: Optional[Unit] = None) -> np.ndarray:
        out = self.original.copy()
        for unit in self.units:
            if unit is skip or not unit.attempts:
                continue
            attempt = unit.current
            x1, y1, x2, y2 = unit.box
            view = out[y1:y2, x1:x2]
            view[attempt.mask] = attempt.result[attempt.mask]
        return out

    def final_mask(self) -> np.ndarray:
        out = self.mask.copy()
        for unit in self.units:
            x1, y1, x2, y2 = unit.box
            out[y1:y2, x1:x2][unit.base_mask] = False
        for unit in self.units:
            x1, y1, x2, y2 = unit.box
            out[y1:y2, x1:x2] |= unit.current.mask if unit.attempts else unit.base_mask
        return out

    async def initial(self, method: str, fill_flat: bool = False) -> None:
        """First attempt for every unit: the whole page mask, block by block, as the pipeline does.
        With ``fill_flat``, areas on a flat background also get a flat fill (attempt 2), which is
        used unless the program finds more problems in it than in the model's result."""
        self.first_method = method
        if not self.units:
            return
        result = await self.backend.inpaint(method, self.original, self.mask.astype(np.uint8) * 255)
        for unit in self.units:
            x1, y1, x2, y2 = unit.box
            unit.attempts = [Attempt(1, method, unit.base_mask.copy(), result[y1:y2, x1:x2].copy(), [])]
            unit.chosen = 0
            unit.work = unit.base_mask.copy()
        await self.check(self.units)
        if not fill_flat or method == FILL:
            return
        filled = []
        for unit in self.units:
            fill = flat_fill(self.crop(unit), unit.base_mask)
            if fill is None:
                continue
            patch, mask, note = fill
            unit.attempts.append(Attempt(2, FILL, mask, patch, [], note=note))
            unit.chosen = 1
            filled.append(unit)
        if not filled:
            return
        await self.check(filled)
        for unit in filled:
            model, fill = unit.attempts
            if len(fill.residual) + len(fill.broken) > len(model.residual) + len(model.broken):
                unit.chosen = 0
            else:
                unit.auto_fill = True

    async def retry(self, unit: Unit, method: str) -> Attempt:
        if unit.work is None or not unit.work.any():
            raise InpaintError(f"#{unit.id}: the mask is empty; undo or add to it first")
        base = self.composite(skip=unit)
        x1, y1, x2, y2 = unit.box
        note = ""
        if method == FILL:
            patch, note = fill_patch(base[y1:y2, x1:x2], unit.work)
        else:
            full = np.zeros(self.mask.shape, np.uint8)
            full[y1:y2, x1:x2][unit.work] = 255
            patch = (await self.backend.inpaint(method, base, full))[y1:y2, x1:x2]
        attempt = Attempt(len(unit.attempts) + 1, method, unit.work.copy(), np.ascontiguousarray(patch).copy(),
                          list(unit.work_ops), note=note)
        unit.attempts.append(attempt)
        unit.chosen = len(unit.attempts) - 1
        unit.retries += 1
        unit.history.clear()
        await self.check([unit])
        return attempt

    def pick(self, unit: Unit, n: int) -> Attempt:
        if not 1 <= int(n) <= len(unit.attempts):
            raise InpaintError(f"#{unit.id} has attempts 1-{len(unit.attempts)}")
        unit.chosen = int(n) - 1
        unit.work = unit.current.mask.copy()
        unit.work_ops = list(unit.current.ops)
        unit.history.clear()
        return unit.current

    async def check(self, units: Sequence[Unit]) -> None:
        page = self.composite()
        text_map = await self.backend.text_map(page)
        if text_map is not None and text_map.shape[:2] != page.shape[:2]:
            text_map = cv2.resize(text_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)
        for unit in units:
            attempt = unit.current
            x1, y1, x2, y2 = unit.box
            mask = attempt.mask.astype(np.uint8)
            if text_map is not None:
                _pixels, _ratio, boxes = qc.residual_text(text_map[y1:y2, x1:x2], mask)
                attempt.residual = [(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1) for bx1, by1, bx2, by2 in boxes]
            _n, total, boxes = qc.broken_lines(self.original[y1:y2, x1:x2], page[y1:y2, x1:x2], mask)
            attempt.broken = [(bx1 + x1, by1 + y1, bx2 + x1, by2 + y1) for bx1, by1, bx2, by2 in boxes]
            attempt.entries = total
            attempt.checked = True

    # --- precise mask operations ---------------------------------------------------
    def _edit(self, unit: Unit, new: np.ndarray, label: str) -> Tuple[int, int]:
        """Text kept as drawn never goes into the mask (``kept_out``: pixels of it this edit reached)."""
        kept = self.kept(unit)
        self.kept_out = int((new & kept & ~unit.work).sum())
        if self.kept_out:
            new = new & ~kept
        before = int(unit.work.sum())
        unit.history.append((unit.work.copy(), list(unit.work_ops)))
        unit.work = new
        unit.work_ops.append(label)
        return before, int(new.sum())

    def undo(self, unit: Unit) -> str:
        if not unit.history:
            raise InpaintError(f"#{unit.id}: no mask edit to undo since the last attempt")
        label = unit.work_ops[-1] if unit.work_ops else "edit"
        unit.work, unit.work_ops = unit.history.pop()
        return label

    def components(self, unit: Unit) -> Tuple[List[Dict], List[Dict]]:
        """Mask components (M1..) and stroke candidates (S1..): ink near the mask that it does not cover."""
        crop = self.crop(unit)
        mask_labels, mask_comps = _components(unit.work)
        for number, comp in enumerate(mask_comps, 1):
            comp["name"] = f"M{number}"
            comp["pixels"] = mask_labels == comp["label"]
        # text of other areas is theirs to erase, and text kept as drawn stays: neither is a candidate
        taken = self.others_mask(unit) | self.kept(unit)
        strokes = ink(crop, unit.work) & ~(cv2.dilate(taken.astype(np.uint8), _kernel(2)) > 0)
        near = cv2.dilate(unit.work.astype(np.uint8), _kernel(max(6, min(16, self._char_size(unit) // 3)))) > 0
        stroke_labels, stroke_comps = _components(strokes)
        candidates = []
        for comp in stroke_comps:
            pixels = stroke_labels == comp["label"]
            area = comp["area"]
            covered = int((pixels & unit.work).sum())
            if area < 4 or covered > 0.6 * area or not comp["textlike"] or not (pixels & near).any():
                continue
            comp["pixels"] = pixels
            candidates.append(comp)
        for number, comp in enumerate(candidates, 1):
            comp["name"] = f"S{number}"
        return mask_comps, candidates

    def _char_size(self, unit: Unit) -> int:
        _labels, comps = _components(unit.base_mask)
        sizes = [min(c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]) for c in comps]
        return int(np.median(sizes)) if sizes else 24

    def select(self, unit: Unit, add: Sequence[int], remove: Sequence[int]) -> Tuple[str, int, int]:
        mask_comps, strokes = self.components(unit)
        new = unit.work.copy()
        done = []
        for number in add or []:
            if not 1 <= int(number) <= len(strokes):
                raise InpaintError(f"S{number} does not exist (S1-S{len(strokes)}); view_region with "
                                   "show=components to see the numbers")
            new |= cv2.dilate(strokes[int(number) - 1]["pixels"].astype(np.uint8), _kernel(2)) > 0
            done.append(f"+S{number}")
        for number in remove or []:
            if not 1 <= int(number) <= len(mask_comps):
                raise InpaintError(f"M{number} does not exist (M1-M{len(mask_comps)})")
            new &= ~mask_comps[int(number) - 1]["pixels"]
            done.append(f"-M{number}")
        if not done:
            raise InpaintError("give stroke numbers to add and/or mask component numbers to remove")
        label = "select " + " ".join(done)
        before, after = self._edit(unit, new, label)
        return label, before, after

    def adjust(self, unit: Unit, op: str, px: int, scope: str = "all",
               components: Sequence[int] = ()) -> Tuple[str, int, int]:
        px = int(px)
        if not 1 <= px <= 12:
            raise InpaintError("px must be 1-12")
        work = unit.work.astype(np.uint8)
        if op == "dilate":
            changed = cv2.dilate(work, _kernel(px)) > 0
        elif op == "erode":
            changed = cv2.erode(work, _kernel(px)) > 0
        else:
            raise InpaintError("op must be dilate or erode")
        if scope == "all":
            zone = np.ones(work.shape, bool)
        elif scope == "residual":
            if not unit.current.residual:
                raise InpaintError(f"#{unit.id} has no residual text marked; use scope all or components")
            zone = np.zeros(work.shape, bool)
            x0, y0 = unit.box[:2]
            for bx1, by1, bx2, by2 in unit.current.residual:
                m = px + 4
                zone[max(0, by1 - y0 - m):max(0, by2 - y0 + m), max(0, bx1 - x0 - m):max(0, bx2 - x0 + m)] = True
        elif scope == "components":
            mask_comps, _strokes = self.components(unit)
            zone = np.zeros(work.shape, bool)
            for number in components or []:
                if not 1 <= int(number) <= len(mask_comps):
                    raise InpaintError(f"M{number} does not exist (M1-M{len(mask_comps)})")
                zone |= cv2.dilate(mask_comps[int(number) - 1]["pixels"].astype(np.uint8), _kernel(px + 1)) > 0
            if not zone.any():
                raise InpaintError("scope components needs component numbers")
        else:
            raise InpaintError("scope must be all, residual or components")
        new = np.where(zone, changed, unit.work)
        label = f"{op} {px}px"
        if scope == "residual":
            label += " (residual)"
        elif scope == "components":
            label += " (" + ",".join(f"M{c}" for c in components) + ")"
        before, after = self._edit(unit, new, label)
        return label, before, after

    def paint(self, unit: Unit, cell_box: Tuple[float, float, float, float], mode: str,
              ink_only: bool = True) -> Tuple[str, int, int]:
        """Add or remove the mask in a rectangle of the unit box (from grid cells)."""
        h, w = unit.work.shape
        x1, y1, x2, y2 = (int(round(v)) for v in cell_box)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            raise InpaintError("those cells are outside this region's view")
        zone = np.zeros((h, w), bool)
        zone[y1:y2, x1:x2] = True
        new = unit.work.copy()
        if mode == "add":
            if ink_only:
                strokes = ink(self.crop(unit), unit.work | zone)
                zone &= cv2.dilate(strokes.astype(np.uint8), _kernel(2)) > 0
            new |= zone
        elif mode == "remove":
            new &= ~zone
        else:
            raise InpaintError("mode must be add or remove")
        label = f"paint {mode}"
        before, after = self._edit(unit, new, label)
        return label, before, after

    def protect_lines(self, unit: Unit, width: int = 2) -> Tuple[str, int, int, int]:
        """Take art lines out of the mask: ink that crosses the mask but lies mostly
        outside it (bubble outlines, panel borders, hair) or is too long to be text."""
        width = max(1, min(6, int(width)))
        strokes = ink(self.crop(unit), unit.work)
        labels, comps = _components(strokes)
        band = np.zeros(unit.work.shape, np.uint8)
        lines = 0
        for comp in comps:
            pixels = labels == comp["label"]
            inside = int((pixels & unit.work).sum())
            if inside == 0:
                continue
            if inside < 0.5 * comp["area"] and comp["area"] >= 12 or not comp["textlike"]:
                band |= pixels.astype(np.uint8)
                lines += 1
        if not lines:
            raise InpaintError(f"#{unit.id}: no art line crosses the mask")
        new = unit.work & ~(cv2.dilate(band, _kernel(width)) > 0)
        label = f"protect lines {width}px"
        before, after = self._edit(unit, new, label)
        return label, before, after, lines

    # --- results -------------------------------------------------------------------
    def score(self) -> float:
        """Lowest unit score: residual text left x0.65, broken lines left x0.9, flagged x0.75."""
        score = 1.0
        for unit in self.units:
            if not unit.attempts:
                continue
            s = 1.0
            if unit.current.residual:
                s *= 0.65
            if unit.current.broken:
                s *= 0.9
            if unit.flagged:
                s *= 0.75
            score = min(score, s)
        return round(score, 3)

    def save_candidates(self, folder: str) -> Dict[str, Dict]:
        """Every attempt as a PNG patch plus its mask, and the sidecar entries."""
        os.makedirs(folder, exist_ok=True)
        out: Dict[str, Dict] = {}
        for unit in self.units:
            attempts = []
            for attempt in unit.attempts:
                name = f"r{unit.id}_a{attempt.n}"
                # cv2.imwrite cannot write a path outside the system code page and says so only by
                # returning False: the editor then found no attempt to switch to (review 2026-11-28 B9)
                for file_name, image in ((name + ".png", cv2.cvtColor(attempt.result, cv2.COLOR_RGB2BGR)),
                                         (name + "_mask.png", attempt.mask.astype(np.uint8) * 255)):
                    _write_png(os.path.join(folder, file_name), image)
                attempts.append({"method": attempt.method, "ops": attempt.ops, "verdict": attempt.issues or ["pass"],
                                 "residual": [list(b) for b in attempt.residual],
                                 "broken_lines": [list(b) for b in attempt.broken], "line_entries": attempt.entries,
                                 "note": attempt.note, "file": name + ".png", "mask": name + "_mask.png"})
            out[str(unit.id)] = {"box": list(unit.box), "regions": unit.regions, "extra": unit.extra,
                                 "attempts": attempts, "chosen": unit.chosen + 1, "flagged": unit.flagged,
                                 "auto_fill": unit.auto_fill, **({"note": unit.note} if unit.note else {})}
        return out
