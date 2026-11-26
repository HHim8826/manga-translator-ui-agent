"""Images sent to the model: numbered pages, zoomed crops with a grid, thumbnails.

The model never gives pixel coordinates; it names regions by the numbers drawn
here and, when zooming, cells of the A1-H8 grid.
"""
from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID_COLUMNS = "ABCDEFGH"
GRID_ROWS = 8
MARK_COLOR = (230, 0, 0)
DONE_COLOR = (0, 150, 60)
SKIP_COLOR = (120, 120, 120)


@lru_cache(maxsize=4)
def _font(size: int) -> ImageFont.ImageFont:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in ("msgothic.ttc", "msyh.ttc", "NotoSansMonoCJK-VF.ttf.ttc"):
        path = os.path.join(root, "fonts", name)
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def open_page(path: str) -> Image.Image:
    with open(path, "rb") as fh:
        image = Image.open(io.BytesIO(fh.read()))
        image.load()
    return image.convert("RGB")


def data_url(image: Image.Image, max_side: Optional[int] = None, quality: int = 90) -> str:
    if max_side and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    image.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def page_url(image: Image.Image, max_side: int) -> Tuple[str, str]:
    """A whole page for a model, its longest side at most ``max_side`` (0: full size), and what to
    say about it in the message: "" at full size, else the size shown (boxes stay in page pixels)."""
    if max_side and max(image.size) > max_side:
        shown = image.copy()
        shown.thumbnail((max_side, max_side), Image.LANCZOS)
        return data_url(shown), f"shown at {shown.width}x{shown.height}; boxes are in page pixels"
    return data_url(image), ""


def draw_marks(page: Image.Image, marks: Sequence[Tuple[int, Sequence[int], str]]) -> Image.Image:
    """Page with a numbered box per region. ``marks`` = (id, bbox, state) where
    state is ``pending`` (red), ``done`` (green) or ``skipped`` (gray)."""
    out = page.copy()
    draw = ImageDraw.Draw(out)
    size = max(18, min(32, out.width // 45))
    font = _font(size)
    for rid, (x1, y1, x2, y2), state in marks:
        color = {"done": DONE_COLOR, "skipped": SKIP_COLOR}.get(state, MARK_COLOR)
        draw.rectangle((x1 - 4, y1 - 4, x2 + 4, y2 + 4), outline=color, width=3)
        label = str(rid)
        tw = draw.textlength(label, font=font)
        tx = max(0, x1 - 4)
        ty = y1 - 8 - size if y1 - 8 - size >= 0 else min(out.height - size - 6, y2 + 6)
        draw.rectangle((tx, ty, tx + tw + 8, ty + size + 4), fill=color)
        draw.text((tx + 4, ty + 1), label, fill="white", font=font)
    return out


def zoom(page: Image.Image, bbox: Sequence[int], others: Sequence[Sequence[Tuple[float, float]]] = (),
         scale: float = 3.0, pad: int = 16, mask_others: bool = True, grid: bool = True,
         max_side: int = 1600) -> Tuple[Image.Image, Dict]:
    """Magnified crop of ``bbox``. Other regions' text inside the crop is grayed
    out (M0.5: otherwise the model reads the neighbor's text too). Returns the
    image and how grid cells map back to page pixels."""
    x1, y1, x2, y2 = bbox
    left, top = max(0, int(x1) - pad), max(0, int(y1) - pad)
    right, bottom = min(page.width, int(x2) + pad), min(page.height, int(y2) + pad)
    crop = page.crop((left, top, right, bottom))
    if mask_others and others:
        draw = ImageDraw.Draw(crop)
        for poly in others:
            pts = [(px - left, py - top) for px, py in poly]
            draw.polygon(pts, fill=(170, 170, 170))
    scale = max(1.0, min(float(scale), max_side / max(1, max(crop.size))))
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
    info = {"crop": [left, top, right, bottom], "scale": round(scale, 2)}
    if grid:
        draw = ImageDraw.Draw(crop)
        font = _font(max(12, min(22, crop.width // 30)))
        cw, ch = crop.width / len(GRID_COLUMNS), crop.height / GRID_ROWS
        for i in range(1, len(GRID_COLUMNS)):
            draw.line([(i * cw, 0), (i * cw, crop.height)], fill=(0, 140, 255), width=1)
        for j in range(1, GRID_ROWS):
            draw.line([(0, j * ch), (crop.width, j * ch)], fill=(0, 140, 255), width=1)
        for i, col in enumerate(GRID_COLUMNS):
            for j in range(GRID_ROWS):
                draw.text((i * cw + 2, j * ch + 1), f"{col}{j + 1}", fill=(0, 110, 230), font=font)
        info["grid"] = {"columns": GRID_COLUMNS, "rows": GRID_ROWS,
                        "cell_page_px": [round((right - left) / len(GRID_COLUMNS), 1),
                                         round((bottom - top) / GRID_ROWS, 1)]}
    return crop, info


def snap_preview(page: Image.Image, bbox: Sequence[int], lines: Sequence[Sequence[Sequence[float]]],
                 erase: np.ndarray, erase_origin: Tuple[int, int],
                 others: Sequence[Sequence[Tuple[float, float]]] = (), pad: int = 40,
                 max_side: int = 1400) -> Image.Image:
    """Crop around a region added by snapping: the pixels that will be erased
    tinted red, its box in green, other regions' boxes in gray."""
    x1, y1, x2, y2 = bbox
    left, top = max(0, int(x1) - pad), max(0, int(y1) - pad)
    right, bottom = min(page.width, int(x2) + pad), min(page.height, int(y2) + pad)
    crop = np.asarray(page.crop((left, top, right, bottom))).astype(np.float32)
    ox, oy = erase_origin
    eh, ew = erase.shape
    sx1, sy1 = max(ox, left), max(oy, top)
    sx2, sy2 = min(ox + ew, right), min(oy + eh, bottom)
    if sx2 > sx1 and sy2 > sy1:
        sel = erase[sy1 - oy:sy2 - oy, sx1 - ox:sx2 - ox] > 0
        view = crop[sy1 - top:sy2 - top, sx1 - left:sx2 - left]
        view[sel] = view[sel] * 0.45 + np.array([255, 0, 0], np.float32) * 0.55
    out = Image.fromarray(crop.clip(0, 255).astype(np.uint8))
    scale = max(1.0, min(3.0, max_side / max(1, max(out.size))))
    out = out.resize((max(1, int(out.width * scale)), max(1, int(out.height * scale))), Image.LANCZOS)
    draw = ImageDraw.Draw(out)
    for poly in others:
        pts = [((px - left) * scale, (py - top) * scale) for px, py in poly]
        draw.polygon(pts, outline=SKIP_COLOR)
    for quad in lines:
        pts = [((px - left) * scale, (py - top) * scale) for px, py in quad]
        draw.polygon(pts, outline=(255, 150, 0))
    draw.rectangle(((x1 - left) * scale - 3, (y1 - top) * scale - 3, (x2 - left) * scale + 3, (y2 - top) * scale + 3),
                   outline=DONE_COLOR, width=3)
    return out


NOTE_COLOR = (255, 140, 0)


def note_preview(page: Image.Image, kept: Sequence[Sequence[Sequence[float]]],
                 note: Sequence[Sequence[Sequence[float]]], text: str, font_size: int,
                 others: Sequence[Sequence[Tuple[float, float]]] = (), pad: int = 40,
                 max_side: int = 1400) -> Image.Image:
    """Crop around text kept as drawn and its note (``note_region``): the kept lines in green, the
    note's lines in orange with its text written in (upright), other regions' boxes in gray."""
    points = np.concatenate([np.asarray(q, dtype=np.float64).reshape(-1, 2) for q in (*kept, *note)])
    left, top = max(0, int(points[:, 0].min()) - pad), max(0, int(points[:, 1].min()) - pad)
    right, bottom = min(page.width, int(points[:, 0].max()) + pad), min(page.height, int(points[:, 1].max()) + pad)
    out = page.crop((left, top, right, bottom))
    scale = max(1.0, min(3.0, max_side / max(1, max(out.size))))
    out = out.resize((max(1, int(out.width * scale)), max(1, int(out.height * scale))), Image.LANCZOS)
    draw = ImageDraw.Draw(out)

    def at(px, py):
        return (px - left) * scale, (py - top) * scale

    for poly in others:
        draw.polygon([at(px, py) for px, py in poly], outline=SKIP_COLOR)
    for quad in kept:
        draw.polygon([at(px, py) for px, py in quad], outline=DONE_COLOR, width=2)
    font = _font(max(8, int(font_size * scale)))
    for quad, line in zip(note, str(text).split("[BR]")):
        draw.polygon([at(px, py) for px, py in quad], outline=NOTE_COLOR, width=2)
        q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
        tl, tr, br, bl = q
        width, height = np.linalg.norm(tr - tl), np.linalg.norm(bl - tl)
        start, end = ((tl + tr) / 2, (bl + br) / 2) if height > width else ((tl + bl) / 2, (tr + br) / 2)
        chars = [ch for ch in line if not ch.isspace()]
        span = np.linalg.norm(end - start)
        step = (end - start) / max(span, 1.0) * font_size
        first = (start + end) / 2 - step * (len(chars) - 1) / 2
        for i, ch in enumerate(chars):
            x, y = at(*(first + step * i))
            try:
                draw.text((x, y), ch, fill=(0, 0, 0), font=font, anchor="mm", stroke_width=2,
                          stroke_fill=(255, 255, 255))
            except ValueError:  # a bitmap font (no font file found) takes no anchor
                draw.text((x - 3, y - 5), ch, fill=(0, 0, 0), font=font)
    return out


def thumbnail(path: str, max_side: int = 640) -> Image.Image:
    image = open_page(path)
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    return image


def side_by_side(images: List[Image.Image], labels: List[str], gap: int = 12) -> Image.Image:
    height = max(im.height for im in images) + 30
    width = sum(im.width for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    font = _font(20)
    x = 0
    for im, label in zip(images, labels):
        draw.text((x + 4, 4), label, fill=(0, 0, 0), font=font)
        out.paste(im, (x, 30))
        x += im.width + gap
    return out
