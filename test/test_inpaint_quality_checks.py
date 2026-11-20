"""Deterministic inpainting checks used before any AI review."""
import cv2
import numpy as np

from manga_translator.inpainting.quality_checks import (
    broken_lines,
    check_inpaint,
    outside_mask_change,
    residual_text,
    summarize,
)


def _white(h=200, w=300):
    return np.full((h, w, 3), 255, np.uint8)


def _rect_mask(shape, x1, y1, x2, y2):
    mask = np.zeros(shape[:2], np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def test_outside_change_ignores_edits_inside_mask():
    original = _white()
    mask = _rect_mask(original.shape, 100, 50, 200, 150)
    result = original.copy()
    result[60:140, 110:190] = 0  # inside the mask
    changed, ratio = outside_mask_change(original, result, mask)
    assert changed == 0 and ratio == 0.0


def test_outside_change_detects_pixels_outside_mask():
    original = _white()
    mask = _rect_mask(original.shape, 100, 50, 200, 150)
    result = original.copy()
    result[5:15, 5:15] = 0
    changed, ratio = outside_mask_change(original, result, mask)
    assert changed == 100 and ratio > 0


def test_outside_change_tolerates_small_differences_and_edge_blending():
    original = _white()
    mask = _rect_mask(original.shape, 100, 50, 200, 150)
    result = original.copy()
    result[:, :] = 250  # within tolerance everywhere
    result[50:150, 98:100] = 0  # blended 2px just outside the mask edge
    assert outside_mask_change(original, result, mask)[0] == 0


def test_residual_text_reports_blobs_inside_mask_only():
    mask = _rect_mask((200, 300), 100, 50, 200, 150)
    text_map = np.zeros((200, 300), np.uint8)
    text_map[80:100, 120:160] = 255  # residual inside the mask
    text_map[10:30, 10:40] = 255  # real text elsewhere on the page
    pixels, ratio, boxes = residual_text(text_map, mask)
    assert pixels == 20 * 40
    assert boxes == [(120, 80, 160, 100)]
    assert 0 < ratio < 1


def test_residual_text_ignores_noise_and_resizes_map():
    mask = _rect_mask((200, 300), 100, 50, 200, 150)
    small_map = np.zeros((100, 150), np.uint8)  # detector map at half size
    small_map[40:42, 70:72] = 255  # 4px at half size -> ~16px, below min_area
    assert residual_text(small_map, mask)[0] == 0


def test_residual_text_ignores_neighbouring_text_the_mask_only_grazes():
    mask = _rect_mask((200, 300), 100, 50, 200, 150)
    text_map = np.zeros((200, 300), np.uint8)
    text_map[140:190, 150:180] = 255  # big sound effect below; 10 of its 50 rows touch the mask
    assert residual_text(text_map, mask)[0] == 0
    text_map[:] = 0
    text_map[135:155, 150:180] = 255  # 60% inside: still the outside glyph's tip
    assert residual_text(text_map, mask)[0] == 0


def test_check_inpaint_flags_one_small_glyph_in_a_large_mask():
    original = _white(400, 600)
    mask = _rect_mask(original.shape, 20, 20, 580, 380)
    text_map = np.zeros(original.shape[:2], np.uint8)
    text_map[200:208, 300:310] = 255  # 80px ghost, 0.04% of the mask
    check = check_inpaint(original, original.copy(), mask, text_map)
    assert check.issues == ["residual_text"] and check.residual_pixels == 80


def _line_page():
    page = _white()
    cv2.line(page, (0, 100), (299, 100), (0, 0, 0), 3)
    return page


def test_broken_lines_passes_when_line_continues_through_mask():
    original = _line_page()
    mask = _rect_mask(original.shape, 100, 60, 200, 140)
    result = original.copy()
    result[60:140, 100:200] = 255
    cv2.line(result, (100, 100), (199, 100), (0, 0, 0), 3)  # repaired line
    broken, total, _ = broken_lines(original, result, mask)
    assert total == 2 and broken == 0


def test_broken_lines_reports_both_entries_when_line_is_erased():
    original = _line_page()
    mask = _rect_mask(original.shape, 100, 60, 200, 140)
    result = original.copy()
    result[60:140, 100:200] = 255  # line wiped out inside the mask
    broken, total, boxes = broken_lines(original, result, mask)
    assert total == 2 and broken == 2 and len(boxes) == 2


def test_broken_lines_has_nothing_to_check_in_a_plain_bubble():
    original = _white()
    cv2.putText(original, "TEXT", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    mask = _rect_mask(original.shape, 100, 60, 220, 140)
    result = _white()
    assert broken_lines(original, result, mask)[:2] == (0, 0)


def test_broken_lines_ignores_a_border_the_mask_runs_along():
    original = _white()
    cv2.rectangle(original, (60, 40), (240, 160), (0, 0, 0), 3)  # bubble border
    cv2.putText(original, "TEXT", (110, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    mask = _rect_mask(original.shape, 63, 43, 238, 158)  # mask hugs the border
    result = _white()
    cv2.rectangle(result, (60, 40), (240, 160), (0, 0, 0), 3)
    assert broken_lines(original, result, mask)[:2] == (0, 0)


def test_broken_lines_ignores_a_line_that_already_ends_at_the_mask():
    original = _white()
    cv2.line(original, (0, 100), (99, 100), (0, 0, 0), 3)  # stops where the mask starts
    mask = _rect_mask(original.shape, 100, 60, 200, 140)
    assert broken_lines(original, original.copy(), mask)[:2] == (0, 0)


def test_broken_lines_counts_a_diagonal_line_entering_the_mask():
    original = _white()
    cv2.line(original, (20, 20), (150, 100), (0, 0, 0), 3)
    mask = _rect_mask(original.shape, 100, 60, 200, 140)
    result = original.copy()
    result[60:140, 100:200] = 255
    broken, total, _ = broken_lines(original, result, mask)
    assert total == 1 and broken == 1


def test_check_inpaint_lists_issue_types():
    original = _line_page()
    mask = _rect_mask(original.shape, 100, 60, 200, 140)
    result = original.copy()
    result[60:140, 100:200] = 255
    result[0:20, 0:20] = 0
    text_map = np.zeros(original.shape[:2], np.uint8)
    text_map[70:90, 120:170] = 255
    check = check_inpaint(original, result, mask, text_map)
    assert set(check.issues) == {"outside_changed", "residual_text", "broken_lines"}
    assert not check.passed

    clean = check_inpaint(original, original.copy(), mask, np.zeros_like(text_map))
    assert clean.passed
    summary = summarize([check, clean])
    assert summary["count"] == 2 and summary["failed"] == 1 and summary["broken_lines"] == 2
