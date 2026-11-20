"""Deterministic checks on an inpainting result.

These run before any AI review (see doc/DESIGN_ai_agent_workflow_2026-11-20.md):
they are cheap, reproducible, and catch the common failures so that only
borderline regions need a model to look at them.

* ``outside_mask_change``  - pixels changed outside the mask (a pipeline bug)
* ``residual_text``        - the text detector still finds text inside the mask
* ``broken_lines``         - art lines that enter the mask but do not continue
* ``check_inpaint``        - all of the above, as one ``InpaintCheck``

All functions take HxW(xC) uint8 arrays and a mask where >0 means "inpainted".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2 (exclusive)


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _kernel(radius: int) -> np.ndarray:
    size = 2 * max(1, radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def outside_mask_change(
    original: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    margin: int = 2,
    tolerance: int = 8,
) -> Tuple[int, float]:
    """Count pixels that changed outside the (slightly dilated) mask.

    ``margin`` allows for blending at the mask edge; ``tolerance`` is the
    largest per-channel difference still treated as unchanged.
    Returns ``(changed_pixels, changed_ratio_of_outside_area)``.
    """
    if original.shape != result.shape:
        raise ValueError(f"shape mismatch: {original.shape} vs {result.shape}")
    allowed = cv2.dilate(_binary(mask), _kernel(margin)) if margin > 0 else _binary(mask)
    outside = allowed == 0
    diff = np.abs(original.astype(np.int16) - result.astype(np.int16))
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    changed = int(np.count_nonzero((diff > tolerance) & outside))
    total = int(np.count_nonzero(outside))
    return changed, (changed / total if total else 0.0)


def residual_text(
    text_map_after: np.ndarray,
    mask: np.ndarray,
    threshold: int = 127,
    min_area: int = 30,
    margin: int = 2,
    min_inside_fraction: float = 0.9,
) -> Tuple[int, float, List[BBox]]:
    """Find text the detector still sees inside the inpainted area.

    ``text_map_after`` is the detector's text map (0-255) for the inpainted
    image, for example ``mask_raw`` from ``detection.dispatch``.
    Returns ``(residual_pixels, residual_ratio_of_mask, component_bboxes)``.
    A detected text component counts only when at least ``min_area`` pixels
    and ``min_inside_fraction`` of it lie inside the mask. Components that
    reach well outside the mask belong to text that was never meant to be
    erased, such as a sound effect the mask edge crosses.
    """
    binary_mask = _binary(mask)
    if text_map_after.shape[:2] != binary_mask.shape:
        text_map_after = cv2.resize(
            text_map_after, binary_mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
    region = cv2.dilate(binary_mask, _kernel(margin)) if margin > 0 else binary_mask
    text = _binary(text_map_after > threshold)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
    boxes: List[BBox] = []
    pixels = 0
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        component = labels[y:y + h, x:x + w] == label
        inside = int(np.count_nonzero(component & (region[y:y + h, x:x + w] > 0)))
        if inside < min_area or inside < area * min_inside_fraction:
            continue
        pixels += inside
        boxes.append((x, y, x + w, y + h))
    mask_area = int(np.count_nonzero(binary_mask))
    return pixels, (pixels / mask_area if mask_area else 0.0), boxes


def _runs_toward_mask(gray: np.ndarray, binary_mask: np.ndarray, max_angle: float) -> np.ndarray:
    """Per pixel: 1 where an edge there would run toward the mask, not along it.

    An edge's image gradient is perpendicular to the edge, so an edge that runs
    within ``max_angle`` degrees of the mask's outward normal has a gradient at
    least ``90 - max_angle`` degrees away from that normal. The normal is the
    gradient of the distance-to-mask map. The image is blurred first so that
    the staircase of an aliased diagonal line still reads as one direction.
    """
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.5)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    distance = cv2.distanceTransform((1 - binary_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    nx = cv2.Sobel(distance, cv2.CV_32F, 1, 0, ksize=3)
    ny = cv2.Sobel(distance, cv2.CV_32F, 0, 1, ksize=3)
    dot = np.abs(gx * nx + gy * ny)
    lengths = np.sqrt(gx * gx + gy * gy) * np.sqrt(nx * nx + ny * ny)
    return (dot <= np.sin(np.radians(max_angle)) * lengths).astype(np.uint8)


def broken_lines(
    original: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    band: int = 3,
    min_entry_pixels: int = 3,
    canny_low: int = 50,
    canny_high: int = 150,
    max_angle: float = 60.0,
) -> Tuple[int, int, List[BBox]]:
    """Find art lines that run into the mask but stop at its edge.

    An "entry" is a group of original edge pixels in the ring just outside
    the mask that touches the mask boundary and runs toward it (within
    ``max_angle`` degrees of the mask's normal). Edges running along the mask
    boundary, such as a bubble border the mask hugs, are not entries, and
    neither are lines whose original already stops at the mask edge (no
    original edges in the strip just inside the mask). An entry is broken when
    the result has no edge pixels in that strip.
    Returns ``(broken_entries, total_entries, broken_entry_bboxes)``.
    This is a signal for review, not proof.
    """
    binary_mask = _binary(mask)
    if not binary_mask.any():
        return 0, 0, []
    k = _kernel(band)
    ring_out = cv2.dilate(binary_mask, k) & (1 - binary_mask)
    # Skip the first 2px inside the mask: a line that just stops at the mask
    # edge still produces an edge (its cut end) right on the boundary.
    ring_in = cv2.erode(binary_mask, _kernel(2)) & (1 - cv2.erode(binary_mask, _kernel(band + 2)))
    boundary_out = cv2.dilate(binary_mask, _kernel(1)) & (1 - binary_mask)

    gray = _gray(original)
    edges_orig = (cv2.Canny(gray, canny_low, canny_high) > 0).astype(np.uint8)
    edges_res = (cv2.Canny(_gray(result), canny_low, canny_high) > 0).astype(np.uint8)

    entries = edges_orig & ring_out & _runs_toward_mask(gray, binary_mask, max_angle)
    # Both sides of one stroke produce an edge; merge them so a line counts once.
    grouped = cv2.dilate(entries, _kernel(2)) & ring_out
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    total = 0
    broken: List[BBox] = []
    for label in range(1, count):
        x, y, w, h, _area = (int(v) for v in stats[label])
        component = (labels == label).astype(np.uint8) & entries
        if np.count_nonzero(component) < min_entry_pixels:
            continue
        if not (component & boundary_out).any():
            continue  # does not actually reach the mask
        reach = cv2.dilate(component, _kernel(band + 3)) & ring_in
        if not (reach & edges_orig).any():
            continue  # the original is blank there too: the line ends at the mask edge
        total += 1
        if not (reach & edges_res).any():
            broken.append((x, y, x + w, y + h))
    return len(broken), total, broken


@dataclass
class InpaintCheck:
    outside_changed_pixels: int = 0
    outside_changed_ratio: float = 0.0
    residual_pixels: Optional[int] = None
    residual_ratio: Optional[float] = None
    residual_boxes: List[BBox] = field(default_factory=list)
    broken_line_count: int = 0
    line_entry_count: int = 0
    broken_line_boxes: List[BBox] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues


def check_inpaint(
    original: np.ndarray,
    result: np.ndarray,
    mask: np.ndarray,
    text_map_after: Optional[np.ndarray] = None,
    *,
    max_outside_ratio: float = 0.0005,
    max_residual_pixels: int = 0,
    max_broken_lines: int = 0,
) -> InpaintCheck:
    """Run every check and list the failures in ``issues``.

    ``issues`` uses the failure types from the design doc:
    ``outside_changed``, ``residual_text``, ``broken_lines``.
    Residual text is only checked when ``text_map_after`` is given. It is
    judged in pixels, not as a share of the mask: one leftover glyph is
    visible however large the rest of the mask is.
    """
    out = InpaintCheck()
    out.outside_changed_pixels, out.outside_changed_ratio = outside_mask_change(original, result, mask)
    if out.outside_changed_ratio > max_outside_ratio:
        out.issues.append("outside_changed")
    if text_map_after is not None:
        out.residual_pixels, out.residual_ratio, out.residual_boxes = residual_text(text_map_after, mask)
        if out.residual_pixels > max_residual_pixels:
            out.issues.append("residual_text")
    out.broken_line_count, out.line_entry_count, out.broken_line_boxes = broken_lines(original, result, mask)
    if out.broken_line_count > max_broken_lines:
        out.issues.append("broken_lines")
    return out


async def detect_text_map(
    image: np.ndarray,
    detector_config,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Run the configured text detector on ``image`` and return its text map.

    ``detector_config`` is ``Config.detector``. YOLO OBB and the SFX filter are
    turned off: the check only needs the pixel-level text map.
    """
    from ..detection import dispatch as dispatch_detection

    result = await dispatch_detection(
        detector_config.detector,
        image,
        detector_config.detection_size,
        detector_config.text_threshold,
        detector_config.box_threshold,
        detector_config.unclip_ratio,
        device,
        False,
        False,
    )
    if not result or len(result) < 2:
        return None
    return result[1]


def summarize(checks: Sequence[InpaintCheck]) -> dict:
    """Aggregate several checks, for reports and A/B comparisons."""
    return {
        "count": len(checks),
        "failed": sum(1 for c in checks if not c.passed),
        "outside_changed": sum(1 for c in checks if "outside_changed" in c.issues),
        "residual_text": sum(1 for c in checks if "residual_text" in c.issues),
        "residual_pixels": sum(c.residual_pixels or 0 for c in checks),
        "broken_lines": sum(c.broken_line_count for c in checks),
        "line_entries": sum(c.line_entry_count for c in checks),
    }
