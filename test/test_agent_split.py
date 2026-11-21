"""split_region: one detected box that holds two separate texts (an aside beside the main text, a
subtitle printed in a logo above the chapter title) becomes two regions, each translated or skipped
where it is; and the invented-script rule shared by the translator and the proofreader."""
import base64
import json
from types import SimpleNamespace

import cv2
import numpy as np

from manga_translator.agent.page import PageDoc
from manga_translator.agent.proofread import Proofreader
from manga_translator.agent.runner import INVENTED_SCRIPT
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentMemory
from manga_translator.agent.tools import AgentSession, ToolRegistry
from test_agent_core import _runner, make_page


def _decode(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_UNCHANGED)


def _page(tmp_path, lines, direction):
    """One region made of ``lines`` (x1, y1, x2, y2), each filled with ink and in the mask."""
    folder = tmp_path / "chapter"
    json_dir = folder / "manga_translator_work" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((240, 320, 3), 255, np.uint8)
    mask = np.zeros((240, 320), np.uint8)
    for x1, y1, x2, y2 in lines:
        image[y1 + 3:y2 - 3, x1 + 3:x2 - 3] = 0
        mask[y1:y2, x1:x2] = 255
    quads = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]] for x1, y1, x2, y2 in lines]
    region = {"lines": quads, "texts": ["main\naside"], "text": "main\naside", "translation": "", "font_size": 24,
              "direction": direction, "prob": 0.9, "render_box_rect_local": [0, 0, 1, 1]}
    image_path = str(folder / "001.png")
    cv2.imwrite(image_path, image)
    ok, png = cv2.imencode(".png", mask)
    data = {image_path: {"regions": [region], "mask_raw": base64.b64encode(png.tobytes()).decode(),
                         "mask_is_refined": True, "original_width": 320, "original_height": 240}}
    json_path = json_dir / "001_translations.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    session = AgentSession(AgentSettings(), AgentMemory(str(tmp_path / "mem")), [image_path])
    session.start_page(0, PageDoc.load(image_path, str(json_path)))
    return session, ToolRegistry(session), json_path


# three columns of the main text and a narrower aside on its left
ASIDE_PAGE = [(220, 20, 250, 200), (180, 20, 210, 200), (140, 20, 170, 160), (100, 60, 118, 180)]


def _saved(json_path):
    return next(iter(json.loads(json_path.read_text(encoding="utf-8")).values()))


def test_an_aside_is_split_off_and_typeset_on_its_own(tmp_path):
    session, tools, json_path = _page(tmp_path, ASIDE_PAGE, "v")
    doc = session.doc
    result = tools.execute("split_region", {"id": 1, "box": [95, 50, 125, 190], "reason": "旁邊的小字是心聲"})
    assert result.ok, result.text
    assert "#1 keeps 3 line(s)" in result.text and "are now #2" in result.text
    main, aside = doc.get(1), doc.get(2)
    assert doc.order == [1, 2] and main.line_count == 3 and aside.line_count == 1
    assert aside.bbox == [100, 60, 118, 180] and main.bbox == [140, 20, 250, 200]
    assert main.font_size == 24 and aside.font_size < main.font_size  # the aside's letters are smaller
    assert aside.direction == "v" and aside.ocr_local == "" and aside.data["text"] == ""
    tools.execute("set_text", {"id": 1, "source_text": "main", "translation": "本文", "category": "dialogue", "confidence": 0.9})
    tools.execute("set_text", {"id": 2, "source_text": "aside", "translation": "心聲", "category": "handwritten", "confidence": 0.9})
    doc.save()
    saved = _saved(json_path)
    assert [r["translation"] for r in saved["regions"]] == ["本文", "心聲"]
    assert [len(r["lines"]) for r in saved["regions"]] == [3, 1]
    assert "render_box_rect_local" not in saved["regions"][0]
    assert doc.sidecar()["regions"][1]["reasons"] == ["旁邊的小字是心聲"]


def test_a_logo_subtitle_is_split_off_and_stays_as_drawn(tmp_path):
    # "ALL UNKNOWN" printed in the logo above "第36話 遺跡発掘", detected as one region of two rows
    session, tools, json_path = _page(tmp_path, [(40, 30, 280, 70), (40, 140, 280, 190)], "h")
    doc = session.doc
    assert tools.execute("split_region", {"id": 1, "box": [30, 20, 290, 80], "reason": "標誌的英文副標"}).ok
    tools.execute("skip_region", {"id": 2, "category": "other", "reason": "標誌的一部分，保留原樣"})
    tools.execute("set_text", {"id": 1, "source_text": "第36話 遺跡発掘", "translation": "第36話　遺跡發掘",
                               "category": "other", "confidence": 0.9})
    doc.save()
    saved = _saved(json_path)
    assert [r["translation"] for r in saved["regions"]] == ["第36話　遺跡發掘"]
    assert len(saved["regions"][0]["lines"]) == 1
    mask = _decode(saved["mask_raw"])
    assert mask[160, 150] == 255 and mask[50, 150] == 0  # the title is erased, the logo's subtitle is not


def test_split_explains_a_box_that_moves_no_line_or_every_line(tmp_path):
    session, tools, _ = _page(tmp_path, ASIDE_PAGE, "v")
    none = tools.execute("split_region", {"id": 1, "box": [0, 0, 60, 60], "reason": "r"})
    assert not none.ok and "holds none of the lines of #1" in none.text and "line 4 [100, 60, 118, 180]" in none.text
    every = tools.execute("split_region", {"id": 1, "box": [0, 0, 320, 240], "reason": "r"})
    assert not every.ok and "holds all of the lines" in every.text
    assert not tools.execute("split_region", {"id": 1, "reason": "r"}).ok  # no box or cells
    assert session.doc.order == [1]  # nothing changed


def test_a_one_line_region_cannot_be_split(tmp_path):
    session, tools, _ = _page(tmp_path, [(40, 30, 280, 70)], "h")
    result = tools.execute("split_region", {"id": 1, "box": [30, 20, 150, 80], "reason": "r"})
    assert not result.ok and "one detected line" in result.text


def test_split_undoes_and_takes_cells_of_a_zoom(tmp_path):
    session, tools, _ = _page(tmp_path, ASIDE_PAGE, "v")
    doc = session.doc
    tools.execute("set_text", {"id": 1, "source_text": "main aside", "translation": "全部", "category": "dialogue", "confidence": 0.9})
    tools.execute("split_region", {"id": 1, "box": [95, 50, 125, 190], "reason": "r"})
    assert doc.get(1).translation is None  # the old text covered both parts
    assert tools.execute("undo", {}).summary == "split #1"
    assert doc.order == [1] and 2 not in doc.regions and doc.get(1).line_count == 4
    assert doc.get(1).translation == "全部"
    assert tools.execute("zoom", {"box": [90, 10, 130, 210]}).ok
    assert tools.execute("split_region", {"id": 1, "cells": "A1:H8", "reason": "r"}).ok
    assert doc.get(2).line_count == 1


def test_the_prompts_teach_split_and_the_invented_script_rule(tmp_path):
    runner, client, store, events = _runner(tmp_path, [make_page(tmp_path, "001")], [])
    events.close()
    prompt = runner.system_prompt()
    assert "split_region them apart before saving" in prompt and INVENTED_SCRIPT in prompt
    assert "split_region" in [t["function"]["name"] for t in runner.registry.schemas()]
    proof = Proofreader.system_prompt(SimpleNamespace(target_lang="CHT", line_breaks=""))
    assert INVENTED_SCRIPT in proof and "do not add the readings back" in proof
    assert "translate the meaning" in INVENTED_SCRIPT and "not what they say" in INVENTED_SCRIPT
