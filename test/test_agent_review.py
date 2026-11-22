"""The editor's AI Agent tab: the review lists from a page's sidecar, finding each flag's box in the
editor, putting a skipped box back and switching an inpaint attempt."""
import json
import os

import cv2
import numpy as np

from test_agents_page import _t, app  # noqa: F401  (one QApplication for the session)
from test_agent_core import _decode, _mask_b64, _quad, make_page

from manga_translator.agent import review
from manga_translator.agent.page import PageDoc
from manga_translator.agent.store import AgentStore
from manga_translator.utils.path_manager import get_inpainted_path, get_work_dir

BOX = [10, 10, 70, 130]  # the inpaint area around region 1 (20, 20, 60, 120)


def _agent_page(tmp_path, units=None):
    """A page the AI Agent finished: regions 1 and 2 translated, 3 skipped (a sign), one flag of
    each agent, and an inpaint area with two attempts."""
    image_path, json_path = make_page(tmp_path)
    doc = PageDoc.load(image_path, json_path)
    doc.set_text(1, "a", "甲", "dialogue", 0.9)
    doc.set_text(2, "b", "乙", "dialogue", 0.9)
    doc.skip(3, "sign", "background sign")
    doc.flag(1, "uncertain_ocr", "handwritten")
    doc.save()
    side = doc.sidecar()
    side["review_flags"] += [
        {"region": 2, "type": "uncertain_translation", "reason": "pun", "agent": "proofread"},
        {"region": 1, "type": "inpaint", "reason": "residue"},  # from before flags named their agent
        {"type": "low_confidence", "reason": "whole page", "agent": "inpaint"},
    ]
    side["inpaint"] = {"units": units or {"1": {
        "box": BOX, "chosen": 1, "flagged": True,
        "attempts": [{"method": "lama", "verdict": ["residual_text"], "file": "r1_a1.png", "mask": "r1_a1_mask.png"},
                     {"method": "fill", "verdict": ["pass"], "file": "r1_a2.png", "mask": "r1_a2_mask.png"}]}}}
    store = AgentStore(get_work_dir(image_path))
    store.save_sidecar("001", side)
    regions = next(iter(json.load(open(json_path, encoding="utf-8")).values()))["regions"]
    return image_path, json_path, store, regions


def test_load_sorts_the_flags_by_agent_and_lists_skipped_boxes_and_areas(tmp_path):
    image_path, _, store, regions = _agent_page(tmp_path)
    page = review.load(image_path)
    assert page.status == "pending" and page.sidecar_path == store.sidecar_path("001")
    assert [(f.agent, f.type, f.number) for f in page.flags] == [
        ("translator", "uncertain_ocr", 1), ("proofread", "uncertain_translation", 2), ("inpaint", "inpaint", 1),
        ("inpaint", "low_confidence", None)]
    translator, proofread, inpaint, whole = page.flags
    assert translator.bbox == [20, 20, 60, 120] and translator.center == [40.0, 70.0]
    assert proofread.json_index == 1 and inpaint.bbox == BOX and whole.bbox is None

    [sign] = page.skipped
    assert (sign.id, sign.category, sign.text, sign.reason) == (3, "sign", "text3", "background sign")
    assert sign.bbox == [200, 150, 280, 180] and sign.data["lines"] == _quad(200, 150, 280, 180)

    [unit] = page.units
    assert (unit.id, unit.box, unit.chosen, unit.flagged) == ("1", BOX, 1, True)
    assert [(a.n, a.method, a.verdict) for a in unit.attempts] == [(1, "lama", ["residual_text"]), (2, "fill", ["pass"])]
    assert unit.attempts[1].file == os.path.join(store.candidates_dir("001"), "r1_a2.png")

    other = tmp_path / "chapter" / "002.png"
    cv2.imwrite(str(other), np.zeros((10, 10, 3), np.uint8))
    assert review.load(str(other)) is None  # the AI Agent never worked on it


def test_flags_find_their_box_in_the_editor(tmp_path):
    image_path, _, _, regions = _agent_page(tmp_path)
    translator, proofread, inpaint, whole = review.load(image_path).flags
    assert len(regions) == 2  # the skipped sign is not in the JSON
    assert review.flag_region(translator, regions) == 0
    assert review.flag_region(proofread, regions) == 1
    assert review.flag_region(inpaint, regions) is None and review.flag_region(whole, regions) is None
    # the human moved region 1 to the front of the list, then deleted it
    moved = [regions[1], regions[0]]
    assert review.flag_region(translator, moved) == 1
    assert review.flag_region(translator, [regions[1]]) is None
    assert review.flag_region(proofread, [regions[1]]) is None


def test_a_skipped_box_comes_back_as_it_was_or_styled_like_its_neighbour(tmp_path):
    image_path, _, _, regions = _agent_page(tmp_path)
    [sign] = review.load(image_path).skipped
    data = review.restore_data(sign, regions)
    assert data["lines"] == _quad(200, 150, 280, 180) and data["font_size"] == 24
    assert (data["text"], data["texts"], data["translation"]) == ("text3", ["text3"], "")
    assert data["center"] == [240.0, 165.0]
    assert sign.data["translation"] == ""  # the sidecar's copy is not touched

    # an older sidecar without the region: a box at its place, styled like the nearest region
    old = review.Skipped(id=9, category="sfx", text="ドン", reason="", center=[150, 60], bbox=[140, 30, 160, 90])
    near = {"lines": _quad(100, 20, 140, 120), "font_family": "Near.ttf", "font_color": "#112233"}
    far = {"lines": _quad(200, 150, 280, 180), "font_family": "Far.ttf"}
    data = review.restore_data(old, [far, near])
    assert data["lines"] == [[[140, 30], [160, 30], [160, 90], [140, 90]]] and data["direction"] == "v"
    assert (data["font_family"], data["font_color"], data["bg_colors"]) == ("Near.ttf", "#112233", [255, 255, 255])
    assert data["text"] == "ドン" and data["translation"] == ""


def _gray(value, shape):
    return np.full((*shape, 3), value, np.uint8)


def _write_attempts(store):
    """Attempt 1 painted the area's top half with 50, attempt 2 a smaller box with 200."""
    folder = store.candidates_dir("001")
    os.makedirs(folder, exist_ok=True)
    h, w = BOX[3] - BOX[1], BOX[2] - BOX[0]
    first = np.zeros((h, w), np.uint8)
    first[:60] = 255
    second = np.zeros((h, w), np.uint8)
    second[20:40, 10:30] = 255
    for n, (value, mask) in enumerate(((50, first), (200, second)), 1):
        cv2.imwrite(os.path.join(folder, f"r1_a{n}.png"), _gray(value, (h, w)))
        cv2.imwrite(os.path.join(folder, f"r1_a{n}_mask.png"), mask)
    return first > 0, second > 0


def test_use_another_inpaint_attempt(tmp_path):
    image_path, json_path, store, _ = _agent_page(tmp_path)
    first, second = _write_attempts(store)
    original = cv2.imread(image_path)
    inpainted = _gray(128, original.shape[:2])
    x1, y1, x2, y2 = BOX
    inpainted[y1:y2, x1:x2][first] = 50  # attempt 1 is in the image now
    target = get_inpainted_path(image_path)
    cv2.imwrite(target, inpainted)
    before_mask = _decode(next(iter(json.load(open(json_path, encoding="utf-8")).values()))["mask_raw"]) > 0

    page = review.load(image_path)
    assert review.use_attempt(page, "1", 2) == target
    result = cv2.imread(target)[y1:y2, x1:x2]
    assert (result[second] == 200).all()
    only_first = first & ~second
    assert (result[only_first] == original[y1:y2, x1:x2][only_first]).all()  # what attempt 1 covered comes back
    assert (result[~first & ~second] == 128).all()
    outside = cv2.imread(target).copy()
    outside[y1:y2, x1:x2] = 128
    assert (outside == 128).all()

    mask = _decode(next(iter(json.load(open(json_path, encoding="utf-8")).values()))["mask_raw"]) > 0
    expected = before_mask.copy()
    expected[y1:y2, x1:x2] = (before_mask[y1:y2, x1:x2] & ~first) | second
    assert (mask == expected).all()
    saved = store.load_sidecar("001")["inpaint"]["units"]["1"]
    assert saved["chosen"] == 2 and saved["picked_in_editor"] is True
    assert page.units[0].chosen == 2 and review.load(image_path).units[0].chosen == 2


def test_an_attempt_that_does_not_fit_changes_nothing(tmp_path):
    import pytest

    image_path, json_path, store, _ = _agent_page(tmp_path)
    page = review.load(image_path)
    with pytest.raises(FileNotFoundError):
        review.use_attempt(page, "1", 2)  # no inpainted image yet
    target = get_inpainted_path(image_path)
    cv2.imwrite(target, _gray(128, (240, 320)))
    with pytest.raises(ValueError):
        review.use_attempt(page, "1", 3)
    folder = store.candidates_dir("001")
    os.makedirs(folder, exist_ok=True)
    for n in (1, 2):
        cv2.imwrite(os.path.join(folder, f"r1_a{n}.png"), _gray(9, (5, 5)))
        cv2.imwrite(os.path.join(folder, f"r1_a{n}_mask.png"), np.full((5, 5), 255, np.uint8))
    with pytest.raises(ValueError, match="does not fit"):
        review.use_attempt(page, "1", 2)
    assert (cv2.imread(target) == 128).all() and store.load_sidecar("001")["inpaint"]["units"]["1"]["chosen"] == 1


def test_the_sidecar_keeps_a_skipped_region_as_it_was_in_the_json(tmp_path):
    image_path, json_path = make_page(tmp_path)
    doc = PageDoc.load(image_path, json_path)
    doc.skip(3, "sfx", "drawn sound")
    side = doc.sidecar()
    kept = [r for r in side["regions"] if r["skipped"]]
    assert [r["id"] for r in kept] == [3] and kept[0]["json"]["lines"] == _quad(200, 150, 280, 180)
    assert all("json" not in r for r in side["regions"] if not r["skipped"])
    json.dumps(side)  # the sidecar stays JSON
    assert side["review_flags"] == []
    doc.flag(2, "layout", "tight")
    assert doc.sidecar()["review_flags"] == [{"region": 2, "type": "layout", "reason": "tight", "agent": "translator"}]


# --- the panel -----------------------------------------------------------------------
def _panel(image_path, regions):
    from ui.editor.agent_review_panel import AgentReviewPanel

    panel = AgentReviewPanel(_t)
    panel.set_page(image_path, regions)
    return panel


def _rows(panel):
    tree = panel.tree
    groups = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    return [(g.text(0), [g.child(i).text(0) for i in range(g.childCount())]) for g in groups]


def _pick(panel, group, row, child=None):
    item = panel.tree.topLevelItem(group).child(row)
    panel.tree.setCurrentItem(item if child is None else item.child(child))


def test_the_panel_lists_what_the_agents_left(app, tmp_path):  # noqa: F811
    image_path, _, _, regions = _agent_page(tmp_path)
    panel = _panel(image_path, regions)
    assert panel.summary_label.text() == "Agent Review Summary state=Agent Page Incomplete"
    kind = "Agent Result With Kind id=Agent Result Editor Box n={} kind=Agent Flag {}".format  # the editor's boxes
    assert _rows(panel) == [
        ("Agent Review Flags count=4", [
            f"Agent Review Agent translator · {kind(1, 'uncertain_ocr')}",
            f"Agent Review Agent proofread · {kind(2, 'uncertain_translation')}",
            "Agent Review Agent inpaint · Agent Review Area id=1",
            "Agent Review Agent inpaint · Agent Flag low_confidence"]),
        ("Agent Review Skipped count=1", ["#3 Agent Skip sign · text3"]),
        ("Agent Review Inpaint count=1", ["Agent Review Unit chosen=1 count=2 id=1"]),
    ]
    unit = panel.tree.topLevelItem(2).child(0)
    assert [unit.child(i).text(0) for i in range(unit.childCount())] == [
        "✓ Agent Review Attempt method=lama n=1 verdict=Agent Issue residual_text",
        "Agent Review Attempt method=Agent Method fill n=2 verdict=Agent Result No Issues"]
    assert panel.detail_label.isHidden()  # nothing picked yet
    # the row stays one line; the reason shows under the list for the picked row
    _pick(panel, 0, 1)
    assert panel.detail_label.text() == f"Agent Review Agent proofread · {kind(2, 'uncertain_translation')}\npun"
    assert not panel.detail_label.isHidden()
    _pick(panel, 2, 0)
    assert panel.detail_label.text() == "Agent Review Unit chosen=1 count=2 id=1\nAgent Review Flagged"
    _pick(panel, 2, 0, 0)
    assert panel.detail_label.text().endswith("\nAgent Review Chosen")

    # the translator's own number is not the editor's: region 1 moved to the end of the list
    panel.set_page(image_path, [regions[1], regions[0]])
    assert _rows(panel)[0][1][0] == f"Agent Review Agent translator · {kind(2, 'uncertain_ocr')}"
    panel.set_page(image_path, [regions[1]])
    assert _rows(panel)[0][1][0] == "Agent Review Agent translator · Agent Flag uncertain_ocr · Agent Review Not Found"

    panel.set_page(str(tmp_path / "chapter" / "009.png"), [])
    assert panel.summary_label.text() == "Agent Review Empty" and panel.tree.topLevelItemCount() == 0


def test_picking_a_row_shows_its_box_restores_and_switches_attempts(app, tmp_path):  # noqa: F811
    image_path, _, _, regions = _agent_page(tmp_path)
    panel = _panel(image_path, regions)
    selected, areas, restored, attempts = [], [], [], []
    panel.region_selected.connect(selected.append)
    panel.area_requested.connect(areas.append)
    panel.restore_requested.connect(restored.append)
    panel.attempt_requested.connect(lambda uid, n: attempts.append((uid, n)))
    assert not panel.restore_button.isEnabled() and not panel.attempt_button.isEnabled()

    _pick(panel, 0, 1)  # the proofreader's flag on the second box
    assert selected == [1] and areas == [[100.0, 20.0, 140.0, 120.0]]
    _pick(panel, 0, 2)  # the inpaint flag: its area, no box to select
    assert selected == [1] and areas[-1] == BOX

    _pick(panel, 1, 0)  # the skipped sign
    assert areas[-1] == [200, 150, 280, 180] and panel.restore_button.isEnabled()
    panel.restore_button.click()
    [data] = restored
    assert data["lines"] == _quad(200, 150, 280, 180) and data["translation"] == ""

    # the editor added it: the row says so and cannot add it twice
    panel.show()
    panel.set_regions(regions + [data])
    assert _rows(panel)[1][1] == ["#3 Agent Skip sign · text3 · Agent Review Restored"]
    assert panel.tree.currentItem().text(0).endswith("Agent Review Restored") and not panel.restore_button.isEnabled()
    panel.hide()
    panel.set_regions(regions)  # hidden: redrawn when the tab shows again
    assert _rows(panel)[1][1] == ["#3 Agent Skip sign · text3 · Agent Review Restored"]
    panel.show()
    assert _rows(panel)[1][1] == ["#3 Agent Skip sign · text3"]
    panel.hide()

    _pick(panel, 2, 0, 0)  # attempt 1 is the one in use
    assert areas[-1] == BOX and not panel.attempt_button.isEnabled()
    _pick(panel, 2, 0, 1)
    assert panel.attempt_button.isEnabled()
    panel.attempt_button.click()
    assert attempts == [("1", 2)]


def test_a_broken_sidecar_leaves_the_panel_empty(app, tmp_path):  # noqa: F811
    image_path, _, store, regions = _agent_page(tmp_path)
    with open(store.sidecar_path("001"), "w", encoding="utf-8") as fh:
        fh.write('{"review_flags": [{"region": "x"}], "inpaint": {"units": {"1": {"box": "bad"}}}}')
    panel = _panel(image_path, regions)
    assert panel.summary_label.text() == "Agent Review Empty"


# --- the editor's side ---------------------------------------------------------------
def _view(**parts):
    import types

    toasts = []
    view = types.SimpleNamespace(_t=_t, _toast=lambda kind, text: toasts.append((kind, text)), **parts)
    return view, toasts


def test_the_editor_reloads_the_tab_for_a_new_page_and_follows_edits(app):  # noqa: F811
    import types

    from editor.region_change import RegionChange
    from ui.editor.view import EditorView

    calls = []
    panel = types.SimpleNamespace(image_path="a.png", set_page=lambda p, r: calls.append(("page", p, len(r))),
                                  set_regions=lambda r: calls.append(("regions", len(r))))
    model = types.SimpleNamespace(get_source_image_path=lambda: "a.png", get_regions=lambda: [{}, {}])
    view, _ = _view(agent_review_panel=panel, model=model)
    EditorView._on_regions_changed_for_agent_review(view, RegionChange.updated([0]))
    EditorView._on_regions_changed_for_agent_review(view, RegionChange.reset())  # reloaded, or a new snapshot
    panel.image_path = "b.png"
    EditorView._on_regions_changed_for_agent_review(view, RegionChange.updated([1]))
    assert calls == [("regions", 2), ("page", "a.png", 2), ("page", "a.png", 2)]


def test_the_editor_centers_restores_and_switches_attempts(app, tmp_path):  # noqa: F811
    import types

    from PyQt6.QtCore import QPointF
    from ui.editor.view import EditorView

    centered = []
    item = types.SimpleNamespace(mapToScene=lambda p: QPointF(p.x() + 1000, p.y()))
    view, toasts = _view(graphics_view=types.SimpleNamespace(_image_item=item, centerOn=centered.append))
    EditorView._show_agent_area(view, [10, 20, 30, 60])
    assert [(p.x(), p.y()) for p in centered] == [(1020.0, 40.0)]

    regions, selections = [{}], []

    def execute(command):
        command.redo()

    model = types.SimpleNamespace(get_regions=lambda: regions,
                                  insert_region=lambda i, r: regions.insert(i, r) or i)
    controller = types.SimpleNamespace(execute_command=execute, set_selection_from_list=selections.append)
    view, toasts = _view(model=model, controller=controller)
    EditorView._restore_agent_region(view, {"text": "看板", "lines": _quad(1, 1, 5, 5)})
    assert regions[-1]["text"] == "看板" and selections == [[1]]
    assert toasts == [("show_info", "Agent Review Restored Hint")]

    image_path, _, store, _ = _agent_page(tmp_path)
    _write_attempts(store)
    cv2.imwrite(get_inpainted_path(image_path), _gray(128, (240, 320)))
    unsaved, loaded = [True], []
    documents = types.SimpleNamespace(_take_prefetched_document=lambda p: loaded.append(("dropped", p)),
                                      do_load_image=lambda p: loaded.append(("load", p)))
    controller = types.SimpleNamespace(export_service=types.SimpleNamespace(has_unsaved_changes=lambda: unsaved[0]),
                                       document_service=documents)
    panel = types.SimpleNamespace(review=review.load(image_path))
    view, toasts = _view(agent_review_panel=panel, controller=controller)
    EditorView._use_agent_attempt(view, "1", 2)
    assert toasts == [("show_info", "Agent Review Save First")] and loaded == []
    unsaved[0] = False
    EditorView._use_agent_attempt(view, "1", 2)
    assert loaded == [("dropped", image_path), ("load", image_path)]
    assert toasts[-1] == ("show_success", "Agent Review Attempt Used n=2")
    assert store.load_sidecar("001")["inpaint"]["units"]["1"]["chosen"] == 2
    EditorView._use_agent_attempt(view, "1", 7)
    assert toasts[-1][0] == "show_error" and len(loaded) == 2


# --- review 14: proofreading and font flags after a reorder --------------------------------------
def test_proofreading_and_font_flags_find_their_box_after_a_reorder(tmp_path):
    image_path, _, store, regions = _agent_page(tmp_path)
    side = store.load_sidecar("001")
    side["review_flags"] = [
        {"region": 2, "type": "uncertain_translation", "reason": "pun", "agent": "proofread",
         "center": [120.0, 70.0], "bbox": [100, 20, 140, 120], "text": "乙"},
        {"region": 1, "type": "layout", "reason": "moved", "agent": "fonts", "bbox": [20, 20, 60, 120]}]
    store.save_sidecar("001", side)
    proofread, fonts = review.load(image_path).flags
    assert proofread.json_index is None and proofread.bbox == [100, 20, 140, 120]
    assert fonts.center == [40.0, 70.0]  # worked out from the box
    assert review.flag_region(proofread, regions) == 1 and review.flag_region(fonts, regions) == 0
    swapped = [regions[1], regions[0]]  # the user moved the second box to the front
    assert review.flag_region(proofread, swapped) == 0 and review.flag_region(fonts, swapped) == 1
    assert review.flag_region(proofread, [regions[0]]) is None  # deleted: not found, not another box


# --- review 11: switching an attempt is all or nothing -------------------------------------------
def test_an_attempt_switch_that_cannot_be_written_leaves_every_file_as_it_was(tmp_path, monkeypatch):
    import pytest

    image_path, json_path, store, _ = _agent_page(tmp_path)
    _write_attempts(store)
    target = get_inpainted_path(image_path)
    cv2.imwrite(target, _gray(128, (240, 320)))
    sidecar = store.sidecar_path("001")
    before = {path: open(path, "rb").read() for path in (target, json_path, sidecar)}
    page = review.load(image_path)
    replace = os.replace

    def failing(src, dst):
        if os.path.abspath(dst) == os.path.abspath(sidecar):
            raise PermissionError("the sidecar is open in another program")
        replace(src, dst)

    monkeypatch.setattr(review.os, "replace", failing)
    with pytest.raises(PermissionError):
        review.use_attempt(page, "1", 2)
    monkeypatch.undo()
    assert {path: open(path, "rb").read() for path in before} == before  # image and JSON put back
    assert page.units[0].chosen == 1
    leftovers = [name for folder in {os.path.dirname(p) for p in before} for name in os.listdir(folder)
                 if name.startswith(".tmp_")]
    assert leftovers == []
    assert review.use_attempt(page, "1", 2) == target  # works once the file can be written
    assert store.load_sidecar("001")["inpaint"]["units"]["1"]["chosen"] == 2


# --- perf: many regions matched at once ----------------------------------------------------------
def test_matching_many_boxes_at_once_finds_what_one_at_a_time_finds():
    """``RegionBoxes`` (a check matches every region of the earlier run) gives ``match_region``'s answer
    for every box: overlapping, nested, touching, equal (the first wins), empty and missing boxes,
    centers on an edge, and the odd values that go the slow way."""
    import random

    rng = random.Random(7)

    def region(x1, y1, x2, y2):
        return {"lines": _quad(x1, y1, x2, y2)}

    regions = []
    for _ in range(120):
        x, y = rng.randrange(0, 900, 10), rng.randrange(0, 1300, 10)
        regions.append(region(x, y, x + rng.choice([0, 10, 40, 80, 200]), y + rng.choice([0, 10, 60, 150])))
    regions += [regions[5], regions[17], {"lines": []}, {"texts": ["no box"]}, region(0, 0, 1000, 1400)]
    boxes = review.RegionBoxes(regions)
    queries = []
    for _ in range(600):
        x, y = rng.randrange(-20, 950, 5), rng.randrange(-20, 1350, 5)
        bbox = [x, y, x + rng.choice([0, 5, 40, 120]), y + rng.choice([0, 5, 60, 160])]
        center = rng.choice([None, [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], [x, y]])
        queries.append((center, bbox))
    for r in regions[:40]:  # each box itself, and its corner as the center
        box = review.region_bbox(r)
        if box:
            queries += [(None, box), ([box[0], box[1]], box), ([box[2], box[3]], [box[2], box[3], box[2] + 1, box[3] + 1])]
    queries += [(None, None), ([1, 2], None), (["x", 1], [0, 0, 10, 10]), (None, [0, 0, float("nan"), 5])]
    def outcome(call):
        try:
            return call()
        except TypeError:  # a text where a number goes: both fail alike
            return "error"

    found = 0
    for center, bbox in queries:
        expected = outcome(lambda: review.match_region(center, bbox, regions))
        got = outcome(lambda: boxes.match(center, bbox))
        assert got == expected, (center, bbox)
        found += got not in (None, "error")
    assert found > 100  # most queries hit a box: the comparison means something

    odd = review.RegionBoxes([region(0, 0, 10, 10), {"lines": [[[0, 0], [float("inf"), 5]]]}])
    assert odd.match(None, [0, 0, 10, 10]) == review.match_region(None, [0, 0, 10, 10], [region(0, 0, 10, 10)])
    assert review.RegionBoxes([]).match([1, 1], [0, 0, 2, 2]) is None
