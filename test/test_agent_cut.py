"""split_region inside one detected line, fit_region, and text kept as drawn next to erased text:
a meaning gloss under invented glyphs, a name on the line of the glyphs, a detected box that took in
some of the art (ch41–43: the translator could not box any of these, the glosses stayed untranslated,
and glyphs kept as drawn were erased with the gloss beside them)."""
import base64
import json

import cv2
import numpy as np

from manga_translator.agent.inpaint import InpaintPage
from manga_translator.agent.page import PageDoc
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentMemory
from manga_translator.agent.tools import AgentSession, ToolRegistry


def _decode(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)


def _page(tmp_path, ink, lines, direction="h", size=(240, 320)):
    """A page with ``ink`` rectangles (x1, y1, x2, y2) drawn black and one region whose detected
    lines are ``lines``; the saved mask covers the lines, as the detector's does."""
    folder = tmp_path / "chapter"
    json_dir = folder / "manga_translator_work" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    height, width = size
    image = np.full((height, width, 3), 255, np.uint8)
    mask = np.zeros((height, width), np.uint8)
    for x1, y1, x2, y2 in ink:
        image[y1:y2, x1:x2] = 0
    image = cv2.GaussianBlur(image, (3, 3), 0)  # soft edges, as scanned lettering has
    for x1, y1, x2, y2 in lines:
        mask[y1:y2, x1:x2] = 255
    quads = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]] for x1, y1, x2, y2 in lines]
    region = {"lines": quads, "texts": ["text"], "text": "text", "translation": "", "font_size": 30,
              "direction": direction, "prob": 0.9}
    image_path = str(folder / "001.png")
    cv2.imwrite(image_path, image)
    ok, png = cv2.imencode(".png", mask)
    data = {image_path: {"regions": [region], "mask_raw": base64.b64encode(png.tobytes()).decode(),
                         "mask_is_refined": True, "original_width": width, "original_height": height}}
    json_path = json_dir / "001_translations.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    session = AgentSession(AgentSettings(), AgentMemory(str(tmp_path / "mem")), [image_path])
    session.start_page(0, PageDoc.load(image_path, str(json_path)))
    return session, ToolRegistry(session), json_path


def _saved(json_path):
    return next(iter(json.loads(json_path.read_text(encoding="utf-8")).values()))


def _near(box, expected, tolerance=2):
    return all(abs(a - b) <= tolerance for a, b in zip(box, expected))


def _glyphs(x, y, count, size=24, gap=10):
    return [(x + i * (size + gap), y, x + i * (size + gap) + size, y + size) for i in range(count)]


# one detected row: four invented glyphs, then a name in kana at the end of the same line
NAME_ROW = _glyphs(50, 40, 4) + _glyphs(200, 44, 2, size=18, gap=8)


def test_a_name_at_the_end_of_a_line_is_cut_out(tmp_path):
    session, tools, json_path = _page(tmp_path, NAME_ROW, [(44, 34, 250, 70)])
    doc = session.doc
    result = tools.execute("split_region", {"id": 1, "box": [194, 36, 248, 68], "reason": "名字寫在字形同一行"})
    assert result.ok, result.text
    assert "Cut out of #1" in result.text and "are now #2" in result.text and result.images
    glyphs, name = doc.get(1), doc.get(2)
    assert doc.order == [1, 2] and name.origin == "cut"
    assert _near(name.bbox, [200, 44, 244, 62])
    assert glyphs.line_count == 1 and glyphs.bbox[2] <= 190  # the line gives up the name
    assert _near(glyphs.bbox, [50, 34, 176, 70]) and glyphs.translation is None  # trimmed to its ink
    tools.execute("skip_region", {"id": 1, "category": "other", "reason": "異世界字形保留"})
    tools.execute("set_text", {"id": 2, "source_text": "りょーた", "translation": "Ryouta", "category": "dialogue",
                               "confidence": 0.9})
    doc.save()
    saved = _saved(json_path)
    assert [r["translation"] for r in saved["regions"]] == ["Ryouta"]
    mask = _decode(saved["mask_raw"])
    assert mask[53, 210] == 255 and mask[53, 235] == 255  # the name is erased
    assert (mask[40:64, 50:180] == 0).all()  # the glyphs stay, also the one next to the name


# a thick detected line: a row of glyphs with a small meaning gloss under the first two
GLOSS_LINE = _glyphs(50, 28, 4) + _glyphs(56, 70, 2, size=14, gap=6)


def test_a_gloss_under_glyphs_is_cut_across_the_line(tmp_path):
    session, tools, json_path = _page(tmp_path, GLOSS_LINE, [(44, 22, 190, 90)])
    doc = session.doc
    assert tools.execute("zoom", {"id": 1}).ok
    result = tools.execute("split_region", {"id": 1, "box": [50, 64, 100, 90], "reason": "「これ」是詞義"})
    assert result.ok, result.text
    glyphs, gloss = doc.get(1), doc.get(2)
    assert _near(gloss.bbox, [56, 70, 90, 84]) and gloss.direction == "h"
    assert _near(glyphs.bbox, [44, 28, 190, 52])  # trimmed across: the gloss row is no longer in it
    tools.execute("skip_region", {"id": 1, "category": "other", "reason": "異世界字形保留"})
    tools.execute("set_text", {"id": 2, "source_text": "これ", "translation": "這個", "category": "dialogue",
                               "confidence": 0.9})
    doc.save()
    mask = _decode(_saved(json_path)["mask_raw"])
    assert mask[77, 62] == 255 and mask[77, 82] == 255
    assert (mask[28:52, 50:180] == 0).all()


def test_the_cut_undoes_in_one_step(tmp_path):
    session, tools, _ = _page(tmp_path, NAME_ROW, [(44, 34, 250, 70)])
    doc = session.doc
    tools.execute("set_text", {"id": 1, "source_text": "x", "translation": "全部", "category": "dialogue",
                               "confidence": 0.9})
    assert tools.execute("split_region", {"id": 1, "box": [194, 36, 248, 68], "reason": "r"}).ok
    assert doc.get(1).translation is None
    assert tools.execute("undo", {}).summary == "split #1"
    assert doc.order == [1] and 2 not in doc.regions and doc.get(1).translation == "全部"
    assert doc.get(1).bbox == [44, 34, 250, 70]


def test_a_cut_that_finds_nothing_or_everything_explains_itself(tmp_path):
    session, tools, _ = _page(tmp_path, NAME_ROW, [(44, 34, 250, 70)])
    empty = tools.execute("split_region", {"id": 1, "box": [10, 150, 60, 200], "reason": "r"})
    assert not empty.ok and "one detected line" in empty.text and "around only the strokes" in empty.text
    every = tools.execute("split_region", {"id": 1, "box": [40, 30, 260, 75], "reason": "r"})
    assert not every.ok and "all of #1's text" in every.text
    assert session.doc.order == [1]


def test_glyphs_at_the_start_of_a_line_are_cut_out(tmp_path):
    session, tools, _ = _page(tmp_path, NAME_ROW, [(44, 34, 250, 70)])
    doc = session.doc
    # add_region cannot take strokes that belong to a region: split_region is the tool for them
    added = tools.execute("add_region", {"box": [194, 36, 248, 68], "direction": "h", "reason": "r"})
    assert not added.ok and "split_region" in added.text
    result = tools.execute("split_region", {"id": 1, "box": [40, 36, 112, 68], "reason": "前兩個字形"})
    assert result.ok, result.text
    assert _near(doc.get(2).bbox, [50, 40, 108, 64])
    assert _near(doc.get(1).bbox, [118, 34, 244, 70])  # the rest of the line, trimmed along it


# the detected box took in a drawn object above the text, so its font size was read too large
ART_AND_TEXT = [(60, 24, 180, 64)] + _glyphs(60, 100, 4, size=20, gap=10)


def test_fit_region_puts_the_box_on_the_text(tmp_path):
    session, tools, json_path = _page(tmp_path, ART_AND_TEXT, [(50, 20, 200, 130)])
    doc = session.doc
    doc.get(1).data["font_size"] = 110
    tools.execute("set_text", {"id": 1, "source_text": "x", "translation": "譯文", "category": "dialogue",
                               "confidence": 0.9})
    result = tools.execute("fit_region", {"id": 1, "box": [52, 92, 196, 128], "reason": "框包含上方的圖"})
    assert result.ok, result.text
    region = doc.get(1)
    assert _near(region.bbox, [60, 100, 170, 120]) and region.line_count == 1
    assert 16 <= region.font_size <= 24 and region.translation == "譯文"
    assert "was 1 line(s) at [50, 20, 200, 130], font≈110px" in result.text
    assert tools.execute("undo", {}).summary == "fit #1" and doc.get(1).font_size == 110
    missing = tools.execute("fit_region", {"id": 1, "box": [200, 150, 300, 230], "reason": "r"})
    assert not missing.ok and "no text strokes" in missing.text


def test_the_tools_and_prompt_teach_the_cut_and_the_fit(tmp_path):
    from test_agent_core import _runner, make_page

    runner, _client, _store, events = _runner(tmp_path, [make_page(tmp_path, "001")], [])
    events.close()
    prompt = runner.system_prompt()
    assert "split_region cuts inside a line too" in prompt and "fit_region it to the text" in prompt
    names = [t["function"]["name"] for t in runner.registry.schemas()]
    assert "fit_region" in names and names.index("fit_region") == names.index("split_region") + 1
    split = next(t for t in runner.registry.schemas() if t["function"]["name"] == "split_region")
    assert "cut out of the line" in split["function"]["description"]


def test_an_inpaint_edit_cannot_take_text_kept_as_drawn():
    image = np.full((60, 120, 3), 255, np.uint8)
    mask = np.zeros((60, 120), np.uint8)
    mask[10:50, 10:110] = 255
    keep = np.zeros((60, 120), bool)
    keep[10:50, 60:110] = True
    quad = np.array([[10, 10], [110, 10], [110, 50], [10, 50]], np.float64)
    page = InpaintPage(image, mask, [(1, [quad], (10, 10, 110, 50))], backend=None, keep=keep)
    assert not (page.mask & keep).any()  # the kept text is out of the mask from the start
    unit = page.units[0]
    page._edit(unit, np.ones_like(unit.work), "select")
    assert page.kept_out > 0 and not (unit.work & page.kept(unit)).any()
