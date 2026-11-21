"""The AI Agent page at the widths the review measured (review 13): the timeline shows the action
and the result of a row without scrolling sideways, and the picked row can be read in full."""
from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)


def _page_at(width, height):
    from PyQt6.QtWidgets import QApplication
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.resize(width, height)
    page.show()
    QApplication.processEvents()
    return page


def _row(page):
    page.replay([
        {"type": "run_start", "run": "r1", "pages": ["001.jpg"], "model": "m", "ts": "2026-11-22T10:00:00+08:00"},
        {"type": "tool_call", "run": "r1", "page": "001", "tool": "set_text", "args": {"id": 3},
         "reason": "the speaker is the older sister, so the line keeps her polite tone", "ts": "2026-11-22T10:00:05+08:00"},
    ])
    return page.timeline.topLevelItem(page.timeline.topLevelItemCount() - 1)


def _visible_from(page, column):
    """How much of a column the timeline shows without scrolling sideways."""
    start = page.timeline.header().sectionViewportPosition(column)
    return page.timeline.viewport().width() - start


def test_the_timeline_keeps_the_action_and_the_result_in_view(app):  # noqa: F811
    from PyQt6.QtCore import Qt

    page = _page_at(800, 600)
    assert page._mode == "narrow" and page.work_splitter.orientation() == Qt.Orientation.Horizontal
    assert page.timeline.isColumnHidden(0)  # the time is in the tooltip and the picked row's text
    assert page.preview.width() <= 260 and page.page_list.width() <= 150  # a slim, tall preview
    assert _visible_from(page, 2) > page.timeline.columnWidth(2) + 100  # the action and the result
    page.close()

    page = _page_at(1200, 720)
    assert page._mode == "wide" and not page.timeline.isColumnHidden(0)
    assert _visible_from(page, 2) > page.timeline.columnWidth(2) + 100
    page.close()


def test_a_very_narrow_page_puts_the_preview_under_the_timeline(app):  # noqa: F811
    from PyQt6.QtCore import Qt

    page = _page_at(1200, 720)
    page._fit_width(700)
    assert page.work_splitter.orientation() == Qt.Orientation.Vertical and page.preview.maximumWidth() > 260
    page._fit_width(1200)  # and back
    assert page.work_splitter.orientation() == Qt.Orientation.Horizontal and page.page_list.maximumWidth() > 150
    page.close()


def test_the_picked_row_is_shown_in_full(app):  # noqa: F811
    page = _page_at(800, 600)
    row = _row(page)
    assert page.row_detail.isHidden()
    page.timeline.setCurrentItem(row)
    detail = page.row_detail.text()
    assert not page.row_detail.isHidden() and row.text(2) in detail and row.text(3) in detail
    assert detail.startswith("10:00:05 · 001 · ")  # the time too, hidden on a narrow page
    assert detail.endswith("\nthe speaker is the older sister, so the line keeps her polite tone")
    page._reset()
    assert page.row_detail.isHidden()
    page.close()
