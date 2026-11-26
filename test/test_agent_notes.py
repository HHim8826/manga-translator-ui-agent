"""Translations set beside text kept as drawn (note_region): placement, the page, the tool and the later stages."""
import json
import math

import cv2
import numpy as np
import pytest

from manga_translator.agent import checks
from manga_translator.agent.inpaint_review import with_kept
from manga_translator.agent.notes import (MIN_NOTE_PX, bracketed, flat_ground, note_font, note_lines, place_in_box,
                                         place_note, usual_size)
from manga_translator.agent.page import NOTED, PageDoc, RegionError
from manga_translator.agent.proofread import ProofPage, ProofSession, _execute
from manga_translator.agent.revise import load_page
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentMemory, AgentStore
from manga_translator.agent.tools import AgentSession, ToolRegistry

from test_agent_core import _decode, make_page


def test_a_note_is_bracketed_once_and_no_line_starts_with_closing_punctuation():
    assert bracketed("路障") == bracketed("（路障）") == bracketed("(路障)") == bracketed("︵路障︶") == "（路障）"
    with pytest.raises(RegionError, match="empty"):
        bracketed(" ")
    lines = note_lines("（物資存放處？）", 4)
    assert "".join(lines) == "（物資存放處？）" and len(lines) == 2
    assert not any(line[0] in "）？" for line in lines)
    assert note_lines("（物資[BR]存放處）", 2) == ["（物資", "存放處）"]


def _column(gray, x1, y1, x2, y2):
    """A column of 'text' strokes."""
    for y in range(y1 + 4, y2 - 12, 20):
        cv2.rectangle(gray, (x1 + 4, y), (x2 - 4, y + 12), 0, 2)
    return [np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)]


def test_a_note_goes_beside_the_text_where_it_covers_the_least_art_and_no_other_text():
    gray = np.full((300, 300), 255, np.uint8)
    poly = _column(gray, 140, 80, 164, 200)
    free = np.zeros_like(gray)
    place = place_note(gray, poly, "v", 0.0, 24, "（路障）", free)
    # left of a column first (read after it), at 0.8 of its size
    assert place.side == "left" and not place.crowded and place.bbox[2] <= 140
    assert place.font_size == 19 and place.direction == "v" and place.text == "（路障）" and len(place.lines) == 1
    # hatching on the left: the right side covers no art
    for x in range(0, 128, 4):
        cv2.line(gray, (x, 0), (x, 299), 0, 1)
    place = place_note(gray, poly, "v", 0.0, 24, "（路障）", free)
    assert place.side == "right" and place.bbox[0] >= 164 and place.ink < 0.05
    # other text on the right: under the column (above it the note would leave the page); the note's
    # column is wider than the text's, with room for its outline
    taken = np.zeros_like(gray)
    taken[:, 172:] = 1
    place = place_note(gray, poly, "v", 0.0, 24, "（路障）", taken)
    assert place.side == "below" and place.bbox[1] >= 200 and not place.crowded
    # nowhere free: the least covered spot, said to be crowded
    place = place_note(gray, poly, "v", 0.0, 24, "（路障）", np.ones_like(gray))
    assert place.crowded


def test_a_note_goes_over_flat_screentone_rather_than_line_art():
    """ch42 p1: the note of 「今ここ」 went over the faces drawn beside it, which have fewer dark pixels
    than the screentone on its other side."""
    gray = np.full((300, 300), 255, np.uint8)
    poly = _column(gray, 140, 10, 164, 290)  # no room above or below it
    for x in range(0, 136, 22):  # line art on the left: a few strokes, little ink
        cv2.line(gray, (x, 0), (x + 18, 299), 0, 2)
        for y in range(20, 300, 40):
            cv2.circle(gray, (x + 10, y), 9, 0, 2)
    gray[:, 170:][(np.indices(gray[:, 170:].shape) % 4 < 2).all(axis=0)] = 0  # dot screentone on the right
    place = place_note(gray, poly, "v", 0.0, 24, "（路障）", np.zeros_like(gray))
    assert place.side == "right" and place.ink > 0.2


def test_a_note_beside_a_slanted_label_runs_along_the_same_slant():
    gray = np.full((400, 400), 255, np.uint8)
    angle = -30.0
    rad = math.radians(angle)
    along, across = np.array([math.cos(rad), math.sin(rad)]), np.array([-math.sin(rad), math.cos(rad)])
    c = np.array([200.0, 200.0])
    quad = np.array([c - along * 80 - across * 12, c + along * 80 - across * 12, c + along * 80 + across * 12,
                     c - along * 80 + across * 12])
    cv2.fillPoly(gray, [np.round(quad).astype(np.int32)], 0)
    place = place_note(gray, [quad], "h", angle, 24, "（帳篷頂）", np.zeros_like(gray))
    assert place.angle == angle and place.side == "below"
    tl, tr, _br, _bl = np.asarray(place.lines[0])
    assert math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0])) == pytest.approx(angle, abs=0.5)
    middle = np.asarray(place.lines[0]).mean(axis=0)
    assert abs(float((middle - c) @ along)) < 1 and float((middle - c) @ across) > 24


def test_a_box_only_says_where_the_note_goes():
    """ch42 p1, the first real run: the model gave a box the size of each label, which made notes of
    12-17 px, and seven calls were refused as too small."""
    gray = np.full((300, 300), 255, np.uint8)
    # the size it takes beside the text: 0.8 of the text's, within 0.6-0.8 of the page's usual size
    assert note_font(10, 30) == 18 and note_font(80, 30) == 24 and note_font(10) == MIN_NOTE_PX
    place = place_in_box(gray, (10, 10, 300, 100), "h", 80, "（路障）", page_px=30)
    assert place.side == "box" and place.font_size == 24 and not place.outgrows
    x1, y1, x2, y2 = place.bbox
    assert abs((x1 + x2) / 2 - 155) <= 1 and abs((y1 + y2) / 2 - 55) <= 1  # centered on the box
    # a box too small for it: the note keeps its size, on one line here, centered on the box and on the page
    small = place_in_box(gray, (10, 10, 40, 20), "h", 24, "（物資存放處？）")
    assert small.font_size == 19 and small.outgrows and small.bbox[0] >= 0 and small.bbox[1] >= 0
    # the fewest lines that fit the box
    tall = place_in_box(gray, (100, 100, 200, 200), "h", 24, "（物資存放處？）")
    assert tall.font_size == 19 and tall.text.count("[BR]") == 1 and not tall.outgrows


def test_the_pages_usual_size_is_its_reading_text():
    # a map page: many small labels, a few bubbles
    assert usual_size([30, 35, 33], [12, 13, 15, 17, 17, 20, 21, 30, 33, 35]) == 33
    assert usual_size([30], [12, 13, 15, 17, 30, 33, 35, 40]) == pytest.approx(33.5)
    assert usual_size([], []) is None


# --- the page and the tool ------------------------------------------------------------
NOTE = {"id": 1, "source_text": "バリケード", "translation": "路障", "category": "handwritten", "confidence": 0.9,
        "reason": "地圖標籤壓在網點和虛線上"}


def _session(tmp_path):
    image_path, json_path = make_page(tmp_path)
    image = cv2.imread(image_path)
    for y in range(0, 130, 14):  # lines running under #1 (20-60 x 20-120), like a map's dashed outlines
        cv2.line(image, (0, y), (88, y + 34), (0, 0, 0), 1)
    # #3 (200-280 x 150-180): letters in a circle its box cuts into
    image[151:180, 201:280] = 255
    for x in range(210, 262, 18):
        cv2.putText(image, "X", (x, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.ellipse(image, (240, 165), (46, 26), 0, 0, 360, (0, 0, 0), 2)
    cv2.imwrite(image_path, image)
    session = AgentSession(AgentSettings(), AgentMemory(str(tmp_path / "mem")), [image_path])
    session.start_page(0, PageDoc.load(image_path, json_path))
    return session, ToolRegistry(session), json_path


def _codes(session, region):
    return [i.code for i in checks.region_issues(session, region)]


def test_note_region_keeps_the_text_and_sets_its_translation_beside_it(tmp_path):
    session, tools, json_path = _session(tmp_path)
    doc = session.doc
    result = tools.execute("note_region", NOTE)
    assert result.ok and result.images and "#1 stays as drawn" in result.text and "is #4" in result.text
    kept, note = doc.regions[1], doc.regions[4]
    assert kept.skipped and kept.skip_category == NOTED and kept.source_text == "バリケード"
    assert note.note_of == 1 and note.translation == "（路障）" and note.origin == "note"
    assert doc.order == [1, 4, 2, 3] and doc.missing() == [2, 3]
    # under #1: no room on its left, and on its right the note would cover #2
    assert note.bbox[1] >= 120 and 20 <= note.bbox[0] and note.bbox[2] <= 60
    lines = session.candidate_lines()
    assert "translated beside it in #4" in lines and "the translation of #1 set beside it" in lines
    # the note is read where its text is drawn
    assert "needs_second_read" in _codes(session, note)
    tools.execute("zoom", {"id": 1})
    assert "needs_second_read" not in _codes(session, note)

    # saved: a region with nothing erased under it; the text it belongs to stays out of the mask
    doc.save()
    page = next(iter(json.load(open(json_path, encoding="utf-8")).values()))
    assert [r["translation"] for r in page["regions"]] == ["（路障）", "", ""]
    saved = page["regions"][0]
    assert saved["text"] == "バリケード" and saved["font_size"] == 19 and saved["direction"] == "v"
    assert saved["stroke_width"] == 0.2  # an outline that reads over screentone
    mask = _decode(page["mask_raw"])
    x1, y1, x2, y2 = note.bbox
    assert mask[60, 40] == 0 and not mask[y1:y2, x1:x2].any()
    entries = {e["id"]: e for e in doc.sidecar()["regions"]}
    assert entries[4]["note_of"] == 1 and entries[1]["skip_category"] == NOTED and "note_of" not in entries[2]

    # text on flat ground (#3 in a circle on white paper) is translated in its place: erasing it breaks
    # nothing. Its box takes in the circle, a line running into it; its strokes alone are on flat ground
    assert not flat_ground(session.page_array, doc.regions[3].polygons)
    flat = tools.execute("note_region", {**NOTE, "id": 3})
    assert not flat.ok and "flat ground" in flat.text and "set_text" in flat.text
    assert not doc.regions[3].skipped and not [r for r in doc.regions.values() if r.note_of == 3]

    # the note changes through the text it belongs to
    refused = tools.execute("set_text", {"id": 4, "source_text": "x", "translation": "y", "category": "dialogue",
                                         "confidence": 0.9})
    assert not refused.ok and "note_region #1 again" in refused.text
    assert "skip_region #1" in tools.execute("skip_region", {"id": 4, "category": "other", "reason": "r"}).text
    assert "undo the note_region" in tools.execute("merge_regions", {"ids": [1, 2], "reason": "r"}).text
    assert "undo the note_region" in tools.execute("fit_region", {"id": 1, "reason": "r"}).text
    # again with a box: the old note goes
    moved = tools.execute("note_region", {**NOTE, "box": [150, 190, 300, 235], "direction": "h"})
    assert moved.ok and "centered on the box you gave" in moved.text
    notes = [r for r in doc.regions.values() if r.note_of == 1]
    assert len(notes) == 1 and notes[0].direction == "h" and notes[0].bbox[1] >= 190
    # translated in its place after all: the note goes; undo brings it back
    tools.execute("set_text", {"id": 1, "source_text": "バリケード", "translation": "路障", "category": "handwritten",
                               "confidence": 0.9})
    assert not [r for r in doc.regions.values() if r.note_of == 1] and not doc.regions[1].skipped
    tools.execute("undo", {})
    assert [r.note_of for r in doc.regions.values() if r.note_of] == [1] and doc.regions[1].skip_category == NOTED


def test_a_page_checked_again_keeps_its_notes_and_the_later_stages_know_them(tmp_path):
    session, tools, json_path = _session(tmp_path)
    assert tools.execute("note_region", NOTE).ok
    for rid in (2, 3):
        tools.execute("set_text", {"id": rid, "source_text": f"text{rid}", "translation": f"譯{rid}",
                                   "category": "dialogue", "confidence": 0.9})
    doc = session.doc
    doc.status = "done"
    doc.save()
    store = AgentStore(str(tmp_path / "chapter" / "manga_translator_work"))
    store.save_sidecar(doc.stem, doc.sidecar())

    again = load_page(doc.image_path, store)
    note = next(r for r in again.regions.values() if r.origin == "note")
    kept = again.regions[note.note_of]
    assert kept.skipped and kept.skip_category == NOTED and note.translation == "（路障）" and again.missing() == []

    # the inpaint review loads the page from its JSON: the note is known, and the text it belongs to is kept
    plain = PageDoc.load(doc.image_path, json_path)
    with_kept(plain, store.load_sidecar(doc.stem))
    assert [r.note_of for r in plain.regions.values() if r.note_of is not None] == [1]
    assert plain.keep_mask()[60, 40]

    # the proofreader lists the text with its note, not as skipped, and a revision keeps the brackets
    page = ProofPage(doc.image_path)
    page.match_sidecar(store.load_sidecar(doc.stem))
    assert page.notes == {1} and not page.skipped and "a note beside the original" in page.listing()[0]
    text, _summary, _images = _execute(ProofSession({page.stem: page}, [], AgentMemory(str(tmp_path / "mem")), "CHT"),
                                       "revise_text", {"page": page.stem, "id": 1, "translation": "柵欄",
                                                       "reason": "地圖上的路障是木柵"})
    assert "revised" in text
    saved = next(iter(json.load(open(json_path, encoding="utf-8")).values()))
    assert saved["regions"][0]["translation"] == "（柵欄）"
