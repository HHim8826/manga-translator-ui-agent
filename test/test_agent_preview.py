"""The Agents page preview: the events carry each page's images and region boxes, and the
panel follows a page from the original to the result."""
import asyncio
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "desktop_qt_ui"))

from manga_translator.agent.events import read_events  # noqa: E402
from manga_translator.agent.page import PageDoc  # noqa: E402
from manga_translator.agent.workflow import run_agent_workflow  # noqa: E402
from manga_translator.utils.path_manager import get_inpainted_path  # noqa: E402

from test_agent_core import FakeTranslator, ScriptedClient, _calls, _finish_page, _runner, make_page  # noqa: E402


# --- events -------------------------------------------------------------------------
def test_page_boxes_follow_the_region_states(tmp_path):
    image_path, json_path = make_page(tmp_path)
    doc = PageDoc.load(image_path, json_path)
    assert [b[5] for b in doc.boxes()] == ["pending"] * 3
    assert doc.boxes()[0][:5] == [1, 20, 20, 60, 120]
    doc.set_text(1, "text1", "譯1")
    doc.skip(2, "sfx", "效果音")
    doc.flag(3, "uncertain_ocr", "看不清")
    assert [b[5] for b in doc.boxes()] == ["done", "skipped", "flagged"]


def test_runner_events_carry_the_image_and_changed_boxes(tmp_path):
    pages = [make_page(tmp_path, "001")]
    script = [_calls(("zoom", {"id": 1})), *_finish_page()]
    runner, _client, _store, events = _runner(tmp_path, pages, script, recall_pass=False)
    asyncio.run(runner.run())
    events.close()
    logged = read_events(events.path)
    start = next(e for e in logged if e["type"] == "page_start")
    assert start["image"] == pages[0][0] and [b[5] for b in start["boxes"]] == ["pending"] * 3
    results = [e for e in logged if e["type"] == "tool_result"]
    assert "boxes" not in results[0]  # zoom changes nothing
    saves = [e for e in results if e["tool"] == "set_text"]
    assert [b[5] for b in saves[0]["boxes"]] == ["done", "pending", "pending"]
    assert [b[5] for b in saves[-1]["boxes"]] == ["done"] * 3
    done = next(e for e in logged if e["type"] == "page_done")
    assert done["image"] == pages[0][0] and [b[5] for b in done["boxes"]] == ["done"] * 3


class RenderingTranslator(FakeTranslator):
    """Saves a result per page in the render call and reports it like MangaTranslator does."""

    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.on_result_saved = None
        self.out = tmp_path / "out"
        self.out.mkdir()

    async def translate_batch(self, images_with_configs, save_info=None):
        result = await super().translate_batch(images_with_configs, save_info)
        if self.load_text:
            for path, _ in images_with_configs:
                output = str(self.out / os.path.basename(path))
                cv2.imwrite(output, np.zeros((10, 10, 3), np.uint8))
                self.on_result_saved(SimpleNamespace(image_name=path, output_path=output))
        return result


def test_render_events_go_to_the_chapter_run_log(tmp_path):
    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / "001.png"), str(folder / "002.png")]

    class Cfg:
        class translator:
            target_lang = "CHT"

    translator = RenderingTranslator(tmp_path)
    client = ScriptedClient(_finish_page() + _finish_page())
    asyncio.run(run_agent_workflow(translator, [(p, Cfg) for p in paths], {}, client=client, log=lambda m: None,
                                   agent_settings={"inpaint_review": False, "recall_pass": False}))
    assert translator.on_result_saved is None  # the hook is removed again
    runs = folder / "manga_translator_work" / "agent" / "runs"
    logged = read_events(str(runs / os.listdir(runs)[0]))
    assert logged[-3]["type"] == "render_start" and logged[-3]["pages"] == 2
    done = [e for e in logged if e["type"] == "render_done"]
    assert [e["page"] for e in done] == ["001", "002"]
    assert done[0]["image"] == paths[0] and done[0]["output"] == str(translator.out / "001.png")
    assert all(e["agent"] == "render" and e["run"] == logged[0]["run"] for e in done)


def test_inpaint_done_names_the_saved_image(tmp_path):
    from test_agent_inpaint import _reviewer

    reviewer, _client, backend, _store, events, path, _json = _reviewer(tmp_path, [_calls(("finish_review", {}))],
                                                                       fill_flat_areas=False)

    def save(image_path, image):
        cv2.imwrite(get_inpainted_path(image_path, create_dir=True), image)

    reviewer.save_image = save
    asyncio.run(reviewer.review_page(path))
    events.close()
    done = next(e for e in read_events(events.path) if e["type"] == "inpaint_done")
    assert done["image"] == path
    assert done["inpainted"] == get_inpainted_path(path, create_dir=False) and os.path.exists(done["inpainted"])


# --- the panel ----------------------------------------------------------------------
_APP = []


@pytest.fixture(scope="module")
def app():
    from PyQt6.QtWidgets import QApplication
    if not _APP:  # kept for the whole session: a second QApplication breaks qfluentwidgets' config object
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def _t(key, **kwargs):
    return key + (" " + " ".join(f"{k}={v}" for k, v in sorted(kwargs.items())) if kwargs else "")


def _images(tmp_path, stem):
    """Original (red), inpainted (green) and result (blue) of one 200x300 page."""
    paths = {}
    for name, bgr in (("image", (0, 0, 255)), ("inpainted", (0, 255, 0)), ("output", (255, 0, 0))):
        path = str(tmp_path / f"{stem}_{name}.png")
        cv2.imwrite(path, np.full((300, 200, 3), bgr, np.uint8))
        paths[name] = path
    return paths


def _color(canvas):
    color = canvas.main.toImage().pixelColor(5, 5)
    return color.red(), color.green(), color.blue()


BOX = [1, 10, 20, 60, 90]


def test_preview_follows_a_page_from_original_to_result(app, tmp_path):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    preview = page.preview
    assert preview.canvas.main is None and preview.canvas.message == "Agents Preview Empty"
    img = _images(tmp_path, "001")
    page.replay([
        {"type": "run_start", "run": "r1", "pages": ["001.jpg"], "model": "m"},
        {"type": "page_start", "run": "r1", "page": "001", "index": 0, "total": 1, "image": img["image"],
         "boxes": [BOX + ["pending"]]},
    ])
    assert preview.shown == "001" and preview.stage == "original" and _color(preview.canvas) == (255, 0, 0)
    assert preview.canvas.boxes == [BOX + ["pending"]] and preview.canvas.source.width() == 200
    assert not preview.stages.widget("result").isEnabled()
    page.replay([
        {"type": "tool_call", "run": "r1", "page": "001", "tool": "set_text", "args": {"id": 1}},
        {"type": "tool_result", "run": "r1", "page": "001", "tool": "set_text", "ok": True, "summary": "#1",
         "boxes": [BOX + ["done"]]},
        {"type": "page_done", "run": "r1", "page": "001", "status": "done", "image": img["image"],
         "boxes": [BOX + ["done"]]},
        {"type": "run_end", "run": "r1", "pages": 1, "done": 1, "flagged": 0, "totals": {"requests": 2}},
    ])
    assert preview.canvas.boxes == [BOX + ["done"]]
    page.handle_event({"type": "inpaint_done", "run": "r1", "page": "001", "agent": "inpaint", "status": "done",
                       "image": img["image"], "inpainted": img["inpainted"]})
    assert preview.stage == "inpainted" and _color(preview.canvas) == (0, 255, 0)
    assert preview.canvas.boxes == []  # the boxes belong on the original
    page.replay([
        {"type": "render_start", "run": "r1", "agent": "render", "pages": 1},
        {"type": "render_done", "run": "r1", "page": "001", "agent": "render", "image": img["image"],
         "output": img["output"], "inpainted": img["inpainted"]},
    ])
    assert preview.stage == "result" and _color(preview.canvas) == (0, 0, 255)
    assert preview.path_label.text() == "001_output.png"
    row = page.timeline.topLevelItem(page.timeline.topLevelItemCount() - 1)
    assert row.text(2) == "Agents Render" and row.text(3) == "Agents Render Detail file=001_output.png"
    assert page.status_label.text().startswith("Agents Finished")
    # comparison: the original left of the divider, the result right of it; a real click on the
    # segment (its signal passes ``checked``, which once replaced the stage name: KeyError True)
    preview.stages.widget("compare").click()
    assert preview.stage == "compare" and not preview.following
    preview.stages.widget("original").click()
    assert preview.stage == "original" and _color(preview.canvas) == (255, 0, 0)
    page.timeline.setCurrentItem(page.timeline.topLevelItem(1))  # a row after a click still renders
    assert preview.shown == "001"
    preview.stages.widget("compare").click()
    assert preview.stage == "compare"
    canvas = preview.canvas
    assert canvas.before is not None and _color(canvas) == (0, 0, 255)
    canvas.resize(400, 600)
    canvas._drag(canvas.image_rect().left() + canvas.image_rect().width() * 0.25)
    assert abs(canvas.split - 0.25) < 0.01


def test_picking_a_page_or_a_row_stops_following(app, tmp_path):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    preview = page.preview
    one, two = _images(tmp_path, "001"), _images(tmp_path, "002")
    page.replay([
        {"type": "run_start", "pages": ["001.jpg", "002.jpg"], "model": "m"},
        {"type": "page_start", "page": "001", "index": 0, "total": 2, "image": one["image"],
         "boxes": [BOX + ["pending"], [2, 100, 100, 150, 200, "pending"]]},
        {"type": "tool_call", "page": "001", "tool": "set_text", "args": {"id": 2}, "reason": "r"},
        {"type": "page_done", "page": "001", "status": "done", "image": one["image"]},
        {"type": "page_start", "page": "002", "index": 1, "total": 2, "image": two["image"], "boxes": []},
    ])
    assert preview.shown == "002"
    # a finished page's inpainting does not pull the view away from the page being translated
    page.handle_event({"type": "inpaint_done", "page": "001", "agent": "inpaint", "inpainted": one["inpainted"]})
    assert preview.shown == "002" and preview.stage == "original"
    page.page_list.setCurrentRow(1)  # 001
    assert preview.shown == "001" and not preview.following and preview.stage == "inpainted"
    page.handle_event({"type": "tool_call", "page": "002", "tool": "zoom", "args": {"id": 1}})
    assert preview.shown == "001"
    # a timeline row of a region: that page's original with the region highlighted
    row = next(page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())
               if page.timeline.topLevelItem(i).text(2) == "Agent Tool set_text")
    page.page_list.setCurrentRow(0)  # all pages: follow again
    assert preview.following and preview.shown == "002"
    page.timeline.setCurrentItem(row)
    assert preview.shown == "001" and preview.highlight == 2 and preview.stage == "original"
    assert not preview.following
    assert preview.canvas.highlight == 2
    preview.follow_box.setChecked(True)
    assert preview.shown == "002"


def test_missing_images_and_other_agents_region_numbers(app, tmp_path):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    preview = page.preview
    page.replay([
        {"type": "run_start", "pages": ["001.jpg"], "model": "m"},
        {"type": "page_start", "page": "001", "index": 0, "total": 1, "image": str(tmp_path / "gone.png"),
         "boxes": []},
        {"type": "tool_call", "page": "001", "agent": "inpaint", "tool": "inpaint_region", "args": {"id": 1}},
    ])
    assert preview.canvas.main is None and preview.canvas.message == "Agents Preview Missing file=gone.png"
    from PyQt6.QtCore import Qt
    row = page.timeline.topLevelItem(0)
    assert row.data(0, Qt.ItemDataRole.UserRole + 2) is None  # inpaint units are not the agent's regions
