"""Review 2026-11-28 B7: the white frame the user set in the editor (``has_custom_white_frame``,
``white_frame_rect_local``) went missing from the JSON after a check or a redo rendered the page
(``TextBlock.to_dict`` wrote only its own fields), and a split or merge kept the old frame on the
new regions (both parts of a split were pulled to the same frame)."""
import json

import numpy as np

from manga_translator.agent.page import PageDoc
from manga_translator.utils.textblock import TextBlock

from test_agent_core import _quad, make_page

FRAME = {"has_custom_white_frame": True, "white_frame_rect_local": [-30.0, -50.0, 30.0, 50.0]}


def test_a_render_writes_back_the_users_white_frame():
    lines = np.array(_quad(20, 20, 60, 120), dtype=np.float64)
    saved = TextBlock(lines=lines, texts=["x"], translation="譯", **FRAME).to_dict()
    assert {key: saved[key] for key in FRAME} == FRAME
    assert not set(FRAME) & set(TextBlock(lines=lines, texts=["x"]).to_dict())  # none set: none written


def test_merged_and_split_regions_drop_the_old_frame(tmp_path):
    image_path, json_path = make_page(tmp_path, boxes=((20, 20, 60, 120), (64, 20, 104, 120), (200, 150, 280, 180)))
    raw = json.load(open(json_path, encoding="utf-8"))
    regions = next(iter(raw.values()))["regions"]
    for region in regions[:2]:
        region.update(FRAME)
    regions[2]["lines"] = _quad(200, 150, 235, 180) + _quad(245, 150, 280, 180)  # two lines: split
    regions[2].update(FRAME)
    open(json_path, "w", encoding="utf-8").write(json.dumps(raw))
    doc = PageDoc.load(image_path, json_path)
    kept = doc.merge([1, 2], "one bubble")
    new = doc.split(3, (240, 140, 290, 190), "two labels")
    for region in (kept, doc.regions[3], new):
        assert not set(FRAME) & set(region.data), region.id
