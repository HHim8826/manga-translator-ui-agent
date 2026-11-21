"""The AI Agent page's preview without a mouse or colours (review 17): it takes the keyboard focus,
zooms and moves with keys and buttons, and the box states read from marks and line styles."""
import cv2
import numpy as np

from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)


def _preview(tmp_path, boxes=None):
    from PyQt6.QtWidgets import QApplication
    from ui.main_page.pages.agent_preview import PagePreview

    cv2.imwrite(str(tmp_path / "001.png"), np.full((300, 200, 3), 255, np.uint8))
    preview = PagePreview(_t)
    preview.resize(320, 520)
    preview.show()
    preview.update_page("001", image=str(tmp_path / "001.png"), boxes=boxes or [])
    preview.show_page("001")
    QApplication.processEvents()
    return preview


def test_the_keyboard_zooms_and_moves_the_preview(app, tmp_path):  # noqa: F811
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    preview = _preview(tmp_path)
    canvas = preview.canvas
    assert canvas.focusPolicy() == Qt.FocusPolicy.StrongFocus
    QTest.keyClick(canvas, Qt.Key.Key_Plus)
    assert abs(canvas.zoom - 1.25) < 1e-9
    QTest.keyClick(canvas, Qt.Key.Key_Equal)  # the + key without Shift
    assert abs(canvas.zoom - 1.5625) < 1e-9
    for _ in range(8):  # larger than the canvas both ways
        QTest.keyClick(canvas, Qt.Key.Key_Plus)
    zoom = canvas.zoom
    rect = canvas.image_rect()
    assert rect.width() > canvas.width() and rect.height() > canvas.height()
    QTest.keyClick(canvas, Qt.Key.Key_Right)  # shows more of the right side
    assert canvas.image_rect().left() < rect.left()
    QTest.keyClick(canvas, Qt.Key.Key_Down)
    assert canvas.image_rect().top() < rect.top()
    QTest.keyClick(canvas, Qt.Key.Key_Minus)
    assert abs(canvas.zoom - zoom / 1.25) < 1e-9
    QTest.keyClick(canvas, Qt.Key.Key_0)
    assert canvas.zoom == 1.0
    preview.close()


def test_the_zoom_buttons_are_reachable_and_named(app, tmp_path):  # noqa: F811
    from PyQt6.QtCore import Qt

    preview = _preview(tmp_path)
    buttons = (preview.zoom_out_button, preview.zoom_in_button, preview.zoom_fit_button)
    assert all(b.focusPolicy() == Qt.FocusPolicy.StrongFocus for b in buttons)
    assert [b.accessibleName() for b in buttons] == ["Agents Preview Zoom Out", "Agents Preview Zoom In",
                                                     "Agents Preview Zoom Fit"]
    preview.zoom_in_button.click()
    preview.zoom_in_button.click()
    assert abs(preview.canvas.zoom - 1.5625) < 1e-9
    preview.zoom_out_button.click()
    assert abs(preview.canvas.zoom - 1.25) < 1e-9
    preview.zoom_fit_button.click()
    assert preview.canvas.zoom == 1.0
    assert preview.canvas.accessibleName() == "Agents Preview"
    preview.close()


def test_box_states_differ_without_colour(app, tmp_path):  # noqa: F811
    from ui.main_page.pages.agent_preview import box_label, box_pen

    states = ("pending", "done", "flagged", "skipped")
    labels = [box_label(state, 3) for state in states]
    assert labels == ["○3", "✓3", "!3", "–3"]
    lines = {(box_pen(state).style(), box_pen(state).width()) for state in states}
    assert len(lines) == 4  # a different line for each state, also in grayscale

    boxes = [[1, 10, 10, 50, 50, "pending"], [2, 60, 10, 90, 50, "done"], [3, 10, 60, 50, 90, "done"],
             [4, 60, 60, 90, 90, "flagged"], [5, 10, 100, 50, 140, "skipped"]]
    preview = _preview(tmp_path, boxes)
    assert preview.canvas.accessibleDescription() == \
        "Agents Preview Box Summary done=2 flagged=1 pending=1 skipped=1"
    preview.close()


def test_the_preview_in_grayscale_still_shows_the_states(app, tmp_path):  # noqa: F811
    """Drawn, the four boxes differ in their lines when the colours are gone."""
    from PyQt6.QtWidgets import QApplication

    boxes = [[1, 10, 10, 90, 90, "pending"], [2, 110, 10, 190, 90, "done"],
             [3, 10, 110, 90, 190, "flagged"], [4, 110, 110, 190, 190, "skipped"]]
    preview = _preview(tmp_path, boxes)
    preview.canvas.resize(200, 300)
    QApplication.processEvents()
    image = preview.canvas.grab().toImage()
    rect = preview.canvas.image_rect()
    scale = rect.width() / 200

    def edge(x, y1, y2):
        """Dark pixels along a box's left edge, in gray."""
        px = int(rect.left() + x * scale)
        values = [image.pixelColor(px, int(rect.top() + y * scale)) for y in range(y1 + 15, y2 - 5)]
        return sum(1 for c in values if (c.red() * 299 + c.green() * 587 + c.blue() * 114) / 1000 < 200)

    solid, dotted, dashed = edge(110, 10, 90), edge(10, 10, 90), edge(110, 110, 190)
    assert solid > dashed > 0 and solid > dotted > 0  # gaps in the dotted and dashed lines
    preview.close()
