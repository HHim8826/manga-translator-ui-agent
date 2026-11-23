"""Where the translation goes on the more involved pages (ch42 p1, a map; ch43 p8, lists of questions):
labels written on a slant get a slanted box and a font of their own size, two translated boxes on top
of each other stop the page, and horizontal text aligned to one side keeps that side."""
import _bootstrap  # noqa: F401
import base64
import json
import math

import cv2
import numpy as np

from manga_translator.agent import checks
from manga_translator.agent.page import PageDoc, Region, layout_alignment
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.snap import snap_text
from manga_translator.agent.store import AgentMemory
from manga_translator.agent.tools import AgentSession, ToolRegistry
from manga_translator.config import Config
from manga_translator.rendering import _calc_region_dst_points_for_font, _resolve_region_layout_center
from manga_translator.utils import TextBlock
from test_agents_page import app  # noqa: F401  (the measuring needs Qt: the session's one QApplication)


def _quad(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _box(x1, y1, x2, y2):
    return np.array(_quad(x1, y1, x2, y2), np.int32)


def _glyph(cx, cy, size, angle):
    """A filled square glyph of ``size`` centered at (cx, cy), turned by ``angle`` degrees."""
    return np.round(cv2.boxPoints(((cx, cy), (size, size), angle))).astype(np.int32)


def _slanted_row(x, y, count, size=20, step=26, angle=-30.0):
    """``count`` glyphs along a line at ``angle`` degrees on the page (y down: negative rises)."""
    t = math.radians(angle)
    return [_glyph(x + i * step * math.cos(t), y + i * step * math.sin(t), size, angle) for i in range(count)]


def _bounds(polys, pad=0):
    pts = np.concatenate(polys)
    return [int(pts[:, 0].min()) - pad, int(pts[:, 1].min()) - pad, int(pts[:, 0].max()) + pad,
            int(pts[:, 1].max()) + pad]


def _page(tmp_path, polys, regions, size=(260, 340)):
    """A page with ``polys`` filled black (soft edges, as scans have) and detector ``regions``:
    dicts with ``lines`` (boxes) and optional fields. Text is the minority of the boxes given to
    the snap below, as it is of a box drawn around real lettering."""
    folder = tmp_path / "chapter"
    json_dir = folder / "manga_translator_work" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    height, width = size
    image = np.full((height, width, 3), 255, np.uint8)
    for poly in polys:
        cv2.fillPoly(image, [poly], (0, 0, 0))
    image = cv2.GaussianBlur(image, (3, 3), 0)
    mask = np.zeros((height, width), np.uint8)
    data_regions = []
    for spec in regions:
        boxes = spec["lines"]
        for x1, y1, x2, y2 in boxes:
            mask[y1:y2, x1:x2] = 255
        region = {"lines": [_quad(*b) for b in boxes], "texts": ["text"], "text": "text", "translation": "",
                  "font_size": 30, "direction": "h", "prob": 0.9}
        region.update({k: v for k, v in spec.items() if k != "lines"})
        data_regions.append(region)
    image_path = str(folder / "001.png")
    cv2.imwrite(image_path, image)
    ok, png = cv2.imencode(".png", mask)
    data = {image_path: {"regions": data_regions, "mask_raw": base64.b64encode(png.tobytes()).decode(),
                         "mask_is_refined": True, "original_width": width, "original_height": height}}
    json_path = json_dir / "001_translations.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    session = AgentSession(AgentSettings(), AgentMemory(str(tmp_path / "mem")), [image_path])
    session.start_page(0, PageDoc.load(image_path, str(json_path)))
    return session, ToolRegistry(session), json_path


def _saved(json_path):
    return next(iter(json.loads(json_path.read_text(encoding="utf-8")).values()))["regions"]


def _translate(tools, rid, text="譯文"):
    result = tools.execute("set_text", {"id": rid, "source_text": "原文", "translation": text,
                                        "category": "dialogue", "confidence": 0.9})
    assert result.ok, result.text


CORNER = {"lines": [(10, 10, 40, 40)]}
CORNER_INK = [_box(18, 18, 32, 32)]


# --- labels written on a slant ----------------------------------------------------------------
def test_a_label_on_a_slant_gets_a_slanted_box_and_its_own_font_size(tmp_path):
    label = _slanted_row(80, 190, 4)  # like バリケード along a road on the map of ch42 p1
    session, tools, json_path = _page(tmp_path, CORNER_INK + label, [CORNER])
    result = tools.execute("add_region", {"box": _bounds(label, 8), "direction": "h", "reason": "斜寫的路標"})
    assert result.ok, result.text
    assert "written on a slant (rising 30° to the right" in result.text
    region = session.doc.get(2)
    assert abs(region.data["angle"] + 30) <= 4 and region.line_count == 1
    assert 16 <= region.font_size <= 28  # the glyphs, not the upright box around the line (about 66 px)
    _translate(tools, 2, "路障")
    session.doc.save()
    block = TextBlock(**_saved(json_path)[1])
    assert abs(block.angle + 30) <= 4
    w, h = block.unrotated_size  # turned back by its angle, the line runs along the x axis
    assert w > 3 * h and h <= 30


def test_a_two_glyph_label_on_a_slant_is_found_too(tmp_path):
    label = _slanted_row(120, 150, 2, size=22, step=28, angle=36.0)  # a word on a roof, falling to the right
    session, _tools, _ = _page(tmp_path, CORNER_INK + label, [CORNER])
    snap = snap_text(session.page_array, _bounds(label, 6), "h")
    assert snap is not None and abs(snap.angle - 36) <= 5 and 16 <= snap.font_size <= 30


def test_upright_text_stays_upright(tmp_path):
    row = [_box(60 + i * 34, 120, 84 + i * 34, 144) for i in range(4)]
    word = [_box(200, 200, 218, 214), _box(222, 196, 236, 218)]  # glyphs of uneven height
    column = [_box(280, 40 + i * 30, 304, 64 + i * 30) for i in range(4)]
    session, _tools, _ = _page(tmp_path, CORNER_INK + row + word + column, [CORNER])
    snap = snap_text(session.page_array, _bounds(row, 10), "h")
    assert snap.angle == 0 and abs(snap.font_size - 24) <= 2
    assert snap_text(session.page_array, _bounds(word, 6), "h").angle == 0
    assert snap_text(session.page_array, _bounds(column, 12), "v").angle == 0


def test_fit_region_turns_an_upright_box_around_a_slanted_label(tmp_path):
    label = _slanted_row(80, 190, 4)
    box = _bounds(label, 4)
    session, tools, _ = _page(tmp_path, CORNER_INK + label, [CORNER, {"lines": [tuple(box)], "font_size": 109}])
    _translate(tools, 2, "路障")
    result = tools.execute("fit_region", {"id": 2, "reason": "斜寫的路標，字級被估成整個框"})
    assert result.ok, result.text
    region = session.doc.get(2)
    assert "written on a slant" in result.text and abs(region.data["angle"] + 30) <= 4
    assert region.font_size <= 28 and region.translation == "路障"
    assert tools.execute("undo", {}).summary == "fit #2"
    assert not session.doc.get(2).data.get("angle") and session.doc.get(2).font_size == 109


def test_add_region_exact_takes_only_the_strokes_in_the_box(tmp_path):
    label = [_box(100 + i * 26, 102, 116 + i * 26, 118) for i in range(3)]
    neighbour = [_box(178 + i * 26, 102, 194 + i * 26, 118) for i in range(2)]
    session, tools, _ = _page(tmp_path, CORNER_INK + label + neighbour, [CORNER])
    grown = tools.execute("add_region", {"box": [94, 94, 170, 126], "direction": "h", "reason": "r"})
    assert grown.ok, grown.text
    assert session.doc.get(2).bbox[2] > 180 and "exact: true" in grown.text  # took the neighbour in
    tools.execute("undo", {})
    exact = tools.execute("add_region", {"box": [94, 94, 170, 126], "direction": "h", "exact": True, "reason": "r"})
    assert exact.ok, exact.text
    assert session.doc.get(2).bbox[2] <= 170 and "exact: true" not in exact.text


# --- translated boxes on top of each other ---------------------------------------------------
def test_translated_boxes_on_top_of_each_other_stop_the_page(tmp_path):
    note = [_box(74 + i * 26, 162, 90 + i * 26, 178) for i in range(3)]
    tent = [_box(74 + i * 26, 102, 90 + i * 26, 118) for i in range(2)]
    # the detector's box of the note took in the tent label above it (ch42 p1 #40 and #19)
    session, tools, _ = _page(tmp_path, CORNER_INK + note + tent,
                              [CORNER, {"lines": [(64, 94, 160, 186)]}, {"lines": [(64, 94, 126, 126)]}])
    for rid in (1, 2, 3):
        _translate(tools, rid)
    issues = checks.overlap_issues(session)
    assert [(i.region, i.detail["other"]) for i in issues] == [(3, 2)]
    done = tools.execute("mark_page_done", {"confidence": 0.9})
    assert not done.ok and "#3: its box and #2's overlap (100% of #3's box)" in done.text
    fitted = tools.execute("fit_region", {"id": 2, "box": [64, 150, 160, 190], "reason": "框包進了上方的標籤"})
    assert fitted.ok, fitted.text
    assert checks.overlap_issues(session) == []


def test_overlap_left_at_the_end_goes_to_the_review_list(tmp_path):
    session, tools, _ = _page(tmp_path, CORNER_INK, [CORNER, {"lines": [(60, 60, 200, 200)]},
                                                     {"lines": [(80, 80, 140, 120)]}])
    for rid in (1, 2, 3):
        _translate(tools, rid)
    assert not tools.execute("mark_page_done", {"confidence": 0.9}).ok
    assert tools.execute("mark_page_done", {"confidence": 0.9}).ok
    flags = [f for f in session.doc.flags if f["type"] == "layout"]
    assert flags and flags[0]["region"] == 3 and "#2" in flags[0]["reason"]


# --- horizontal text aligned to one side ------------------------------------------------------
LEFT = [(40, 40, 250, 70), (40, 80, 180, 110), (40, 120, 120, 150)]  # 塩って貴重？ / →はい
CENTERED = [(40, 40, 250, 70), (75, 80, 215, 110), (105, 120, 185, 150)]
RIGHT = [(40, 40, 250, 70), (110, 80, 250, 110), (170, 120, 250, 150)]


def _down(boxes, dy):
    return [(x1, y1 + dy, x2, y2 + dy) for x1, y1, x2, y2 in boxes]


def test_alignment_follows_the_source_lines(tmp_path):
    session, tools, json_path = _page(
        tmp_path, CORNER_INK,
        [CORNER, {"lines": LEFT}, {"lines": _down(CENTERED, 140), "alignment": "center"},
         {"lines": _down(RIGHT, 280)},
         {"lines": [(300, 160, 330, 250), (270, 160, 300, 210)], "direction": "v"}],
        size=(460, 340))
    doc = session.doc
    assert [layout_alignment(doc.get(i)) for i in (1, 2, 3, 4, 5)] == ["", "left", "", "right", ""]
    for rid in (1, 2, 3, 4, 5):
        _translate(tools, rid, "問題？[BR]→是")
    doc.save()
    saved = _saved(json_path)
    assert saved[1]["alignment"] == "left" and saved[3]["alignment"] == "right"
    assert saved[2]["alignment"] == "center" and "alignment" not in saved[4]


def test_an_alignment_set_to_a_side_stays(tmp_path):
    session, tools, json_path = _page(tmp_path, CORNER_INK, [CORNER, {"lines": RIGHT, "alignment": "left"}])
    for rid in (1, 2):
        _translate(tools, rid)
    session.doc.save()
    assert _saved(json_path)[1]["alignment"] == "left"


def _block(alignment, lines=LEFT, direction="h"):
    return TextBlock(lines=[_quad(*b) for b in lines], texts=["x"], font_size=30, direction=direction,
                     alignment=alignment, translation="鹽很珍貴嗎？\n→是", target_lang="CHT")


def _dst(block, mode):
    return _calc_region_dst_points_for_font(block, 30, True, 1.0, 1.0, Config(), anchor_mode=mode)[0]


def test_the_renderer_puts_left_aligned_text_against_the_left_of_its_box(app):
    left, centered = _block("left"), _block("center")
    for mode in ("top", "center"):
        drawn, middle = _dst(left, mode), _dst(centered, mode)
        assert middle[:, 0].max() - middle[:, 0].min() < 200  # narrower than the box (210 px): room to move
        assert abs(drawn[:, 0].min() - 40) <= 1  # starts where the source lines start
        assert abs(drawn[:, 1].min() - middle[:, 1].min()) <= 0.5  # only moved sideways
    assert abs(_dst(_block("right", RIGHT), "top")[:, 0].max() - 250) <= 1
    # the editor's own frame, vertical text and text wider than its box stay where they were
    config = Config()
    assert _resolve_region_layout_center(left, 30, True, 1.0, 1.0, config, anchor_mode="center_box") == \
        tuple(left.center)
    tall = _block("left", [(40, 40, 70, 250), (80, 40, 110, 180)], direction="v")
    assert _resolve_region_layout_center(tall, 30, False, 1.0, 1.0, config, anchor_mode="center") == tuple(tall.center)
    narrow = _block("left", [(40, 40, 120, 70), (40, 80, 90, 110)])
    assert _resolve_region_layout_center(narrow, 30, True, 1.0, 1.0, config, anchor_mode="center") == \
        tuple(narrow.center)


def test_slanted_labels_side_by_side_do_not_overlap(tmp_path):
    road = _slanted_row(70, 150, 4)
    roof = _slanted_row(88, 182, 4)  # the next strip of the map, just below it
    session, tools, _ = _page(tmp_path, CORNER_INK + road + roof, [CORNER])
    for label in (road, roof):
        added = tools.execute("add_region", {"box": _bounds(label, 4), "direction": "h", "exact": True, "reason": "r"})
        assert added.ok, added.text
    a, b = session.doc.get(2), session.doc.get(3)
    upright = max(0, min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])) * \
        max(0, min(a.bbox[3], b.bbox[3]) - max(a.bbox[1], b.bbox[1]))
    assert upright > 0.3 * (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])  # the upright boxes do overlap
    for rid in (1, 2, 3):
        _translate(tools, rid)
    assert checks.overlap_issues(session) == []


def test_an_upright_box_around_a_slanted_label_leaves_the_next_label_its_strokes(tmp_path):
    road = _slanted_row(70, 150, 5)
    roof = _slanted_row(150, 170, 3)  # starts inside the upright box of the road label
    session, _tools, _ = _page(tmp_path, CORNER_INK + road + roof, [CORNER])
    box = _bounds(road, 4)
    for grow in (True, False):
        snap = snap_text(session.page_array, box, "h", grow=grow)
        assert snap.strokes == 5 and abs(snap.angle + 30) <= 4, (grow, snap.strokes, snap.angle)
        ox, oy = snap.erase_origin
        for glyph in roof:
            cx, cy = glyph.mean(axis=0).astype(int)
            inside = 0 <= cy - oy < snap.erase.shape[0] and 0 <= cx - ox < snap.erase.shape[1]
            assert not (inside and snap.erase[cy - oy, cx - ox])  # the roof label is not erased with it


def test_screentone_dots_are_not_taken_for_text(tmp_path):
    tone = [_box(x, y, x + 3, y + 3) for x in range(60, 260, 7) for y in range(90, 170, 7)]
    word = [_box(100 + i * 34, 118, 122 + i * 34, 140) for i in range(3)]
    session, _tools, _ = _page(tmp_path, CORNER_INK + tone, [CORNER])
    # the glyphs are drawn over the tone: white around them as lettering on a toned roof has
    image = session.page_array.copy()
    for x1, y1, x2, y2 in [(96, 114, 126, 144), (130, 114, 160, 144), (164, 114, 194, 144)]:
        image[y1:y2, x1:x2] = 255
    for glyph in word:
        cv2.fillPoly(image, [glyph], (0, 0, 0))
    snap = snap_text(image, [80, 100, 214, 158], "h")
    assert snap.strokes == 3 and abs(snap.font_size - 22) <= 3
    assert snap.erase.sum() < 3 * 3 * 900  # the glyphs and a rim, not the toned area around them


def test_brackets_around_two_lines_do_not_join_them(tmp_path):
    """ch42 p1: a handwritten "（資材 / 置き場？）", its brackets as tall as both lines."""
    glyphs = [_box(100 + i * 22, y, 116 + i * 22, y + 16) for y in (100, 122) for i in range(3)]
    brackets = [_box(90, 96, 94, 142), _box(166, 96, 170, 142)]
    session, _tools, _ = _page(tmp_path, CORNER_INK + glyphs + brackets, [CORNER])
    snap = snap_text(session.page_array, [80, 86, 180, 152], "h")
    assert len(snap.lines) == 2 and abs(snap.font_size - 16) <= 3, (snap.lines, snap.font_size)
    assert snap.strokes == 8  # the brackets are erased with the text


def test_two_lines_of_handwriting_that_touch_are_two_lines(tmp_path):
    """ch42 p1: the lines of "（資材 / 置き場？）" are set so tight that a stroke joins them."""
    glyphs = [_box(100 + i * 22, y, 116 + i * 22, y + 17) for y in (100, 121) for i in range(3)]
    mark = [_box(168, 115, 171, 123)]  # the "？" of the lower line, set as high as the upper one
    session, _tools, _ = _page(tmp_path, CORNER_INK + glyphs + mark, [CORNER])
    snap = snap_text(session.page_array, [86, 88, 182, 150], "h")
    assert len(snap.lines) == 2 and abs(snap.font_size - 17) <= 3, (snap.lines, snap.font_size)


def test_one_line_of_big_glyphs_is_not_cut(tmp_path):
    glyphs = [_box(70 + i * 40, 100, 104 + i * 40, 140) for i in range(3)]
    glyphs += [_box(76 + i * 40, 108, 98 + i * 40, 112) for i in range(3)]  # small strokes as well
    session, _tools, _ = _page(tmp_path, CORNER_INK + glyphs, [CORNER])
    snap = snap_text(session.page_array, [56, 86, 204, 154], "h")
    assert len(snap.lines) == 1, snap.lines


def test_full_width_punctuation_takes_a_whole_character():
    """ch42 p1: 「（物資[BR]存放處？）」 in the box of the two narrow handwritten lines 「（資材 / 置き場？）」 was
    drawn at 12 px for 19 px of source: its brackets and question mark are as wide as the characters."""
    box = Region(id=26, data={"lines": [[[487, 1493], [558, 1493], [558, 1513], [487, 1513]],
                                        [[487, 1513], [558, 1513], [558, 1537], [487, 1537]]],
                              "direction": "horizontal", "font_size": 22})
    assert checks.line_overflow(box, "（物資[BR]存放處？）")
    assert checks.line_overflow(box, "物資[BR]存放處？") is None


def test_a_stroke_belongs_to_the_region_that_holds_most_of_it():
    """ch43 p1: the lower stroke of 「こ」 in the gloss 「これ」 dips into the loose box of the line
    「でぃめーとりあ」 under it. Cut out of the glyph line, the gloss keeps that stroke whole (it was
    clipped and stayed drawn under the translation), and a stroke of the line under it that pokes
    up into the box given for the gloss is not taken (its tip was erased with the gloss)."""
    image = np.full((200, 240, 3), 255, np.uint8)
    gloss = [(40, 96, 44, 114), (52, 98, 66, 102), (52, 116, 66, 121), (76, 96, 80, 118), (84, 100, 96, 104)]
    for x1, y1, x2, y2 in gloss + [(112, 112, 118, 135)]:  # the last one: a stroke of the line under it
        image[y1:y2, x1:x2] = 0
    for x in range(30, 200, 30):  # the line under it, drawn from y 135
        image[135:165, x:x + 22] = 0
    image = cv2.GaussianBlur(image, (3, 3), 0)  # soft edges, as scanned lettering has
    under = np.zeros((200, 240), np.uint8)
    under[120:170, 25:205] = 1  # its detected box starts well above its ink
    exclude = cv2.dilate(under, np.ones((5, 5), np.uint8))  # as PageDoc.occupied grows it
    snap = snap_text(image, (36, 92, 122, 126), "h", exclude=exclude, grow=False)
    assert snap is not None
    assert snap.bbox[3] >= 120 and snap.bbox[2] <= 100  # the whole foot of 「こ」, not the stroke under it
    erase = np.zeros((200, 240), bool)
    ex, ey = snap.erase_origin
    erase[ey:ey + snap.erase.shape[0], ex:ex + snap.erase.shape[1]] = snap.erase > 0
    assert erase[116:121, 52:66].all()
    assert not erase[112:135, 112:118].any()
