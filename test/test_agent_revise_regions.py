"""Review 2026-11-28 B15, a page checked again: style none left the old style's font in the JSON, a
new style alone kept the old font's rich text, translating text kept as drawn dropped the note the
user had changed beside it, and a split or merge of text the earlier run skipped lost that it was
not in the saved mask."""
import copy
import json

import pytest

from manga_translator.agent.page import PageDoc, Region, RegionError

from test_agent_core import _quad, _session, make_page


def _text(i, **extra):
    return json.dumps({"id": i, "source_text": f"text{i}", "translation": f"譯{i}", "category": "narration",
                       "confidence": 0.9, **extra})


def test_style_none_takes_the_old_styles_font_out(tmp_path):
    session, tools = _session(tmp_path)
    region = session.doc.regions[1]
    region.data["font_family"] = "Noto Serif TC"  # the earlier run's style wrote it
    region.font, region.style = "Noto Serif TC", "旁白"
    assert "saved" in tools.execute("set_text", _text(1, style="none")).text
    out = session.doc.to_json_regions()
    assert "font_family" not in out[0]  # the render uses the user's font
    session.doc.regions[2].data["font_family"] = "User Font.ttf"
    tools.execute("set_text", _text(2, style="none"))  # had no style: the JSON's font is the user's
    assert session.doc.to_json_regions()[1]["font_family"] == "User Font.ttf"


def test_a_new_style_alone_is_a_change(tmp_path):
    image_path, json_path = make_page(tmp_path)
    doc = PageDoc.load(image_path, json_path)
    region = doc.regions[1]
    region.source_text, region.translation = "text1", "譯1"
    region.data.update(translation="譯1", translation_rich={"old": "font"})
    region.loaded = copy.deepcopy(region.data)
    region.loaded_state = region.state()
    assert region.unchanged
    region.style, region.font = "旁白", "Noto Serif TC"
    assert not region.unchanged
    out = doc.to_json_regions()[0]
    assert out["font_family"] == "Noto Serif TC" and "translation_rich" not in out


def test_text_with_the_users_note_beside_it_stays(tmp_path):
    image_path, json_path = make_page(tmp_path)
    doc = PageDoc.load(image_path, json_path)
    doc.skip(3, "sign", "kept as drawn with a note")
    note = Region(id=9, data={"lines": _quad(200, 190, 280, 210), "texts": [""]}, translation="招牌", note_of=3,
                  locked=True)
    doc.regions[9] = note
    doc.order.append(9)
    for call in (lambda: doc.set_text(3, "看板", "招牌"), lambda: doc.skip(3, "sign", "again")):
        with pytest.raises(RegionError, match="the user's note #9"):
            call()
    assert 9 in doc.regions


def test_split_and_merge_keep_that_the_earlier_run_skipped_the_text(tmp_path):
    image_path, json_path = make_page(tmp_path, boxes=((20, 20, 60, 120), (64, 20, 104, 120), (200, 150, 280, 180)))
    raw = json.load(open(json_path, encoding="utf-8"))
    next(iter(raw.values()))["regions"][2]["lines"] = _quad(200, 150, 235, 180) + _quad(245, 150, 280, 180)
    open(json_path, "w", encoding="utf-8").write(json.dumps(raw))
    doc = PageDoc.load(image_path, json_path)
    doc.revise = True
    doc.regions[3].was_skipped = True
    new = doc.split(3, (240, 140, 290, 190), "two labels")
    assert new.was_skipped and doc.regions[3].was_skipped
    doc.regions[2].was_skipped = True
    assert doc.merge([1, 2], "one bubble").was_skipped
