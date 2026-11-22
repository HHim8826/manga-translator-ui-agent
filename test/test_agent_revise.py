"""Checking again a chapter an earlier run finished (``manga_translator.agent.revise``): the page as
it was left, the user's changes locked, thin fonts replaced, only the changes inpainted again."""
import asyncio
import json
import os

import cv2
import numpy as np
import pytest

from manga_translator.agent import revise
from manga_translator.agent.fonts import FontFace, FontService
from manga_translator.agent.page import RegionError
from manga_translator.agent.proofread import ProofPage
from manga_translator.agent.revise import inpaint_changes, json_center, load_page, locked_centers, text_key
from manga_translator.agent.store import AgentStore
from manga_translator.utils.path_manager import get_inpainted_path

from test_agent_core import (FakeTranslator, ScriptedClient, _calls, _decode, _mask_b64, _quad, make_page,
                             run_agent_workflow)

BOXES = ((20, 20, 60, 120), (100, 20, 140, 120), (200, 150, 280, 180), (160, 20, 190, 120))


class Cfg:
    class translator:
        target_lang = "CHT"


def _say(i, text):
    return ("set_text", {"id": i, "source_text": f"text{i}", "translation": text, "category": "dialogue",
                         "confidence": 0.9, "reason": "讀圖"})


FIRST_RUN = [_calls(_say(1, "譯1……"), _say(2, "譯2"), _say(4, "譯4"),
                    ("skip_region", {"id": 3, "category": "sign", "reason": "背景招牌"})),
             _calls(("mark_page_done", {"confidence": 0.9}))]


def _first_run(tmp_path, n_pages=1):
    """A chapter the agent finished: page 001 (and 002 ...) with four boxes, #3 skipped."""
    folder = tmp_path / "chapter"
    folder.mkdir(exist_ok=True)
    paths = [str(folder / f"{i:03d}.png") for i in range(1, n_pages + 1)]

    class Preprocess(FakeTranslator):
        async def translate_batch(self, images_with_configs, save_info=None):
            if self.template:
                self.calls.append({"template": True})
                for path, _ in images_with_configs:
                    make_page(self.tmp_path, os.path.splitext(os.path.basename(path))[0], boxes=BOXES)
                return [f"ctx:{os.path.basename(p)}" for p, _ in images_with_configs]
            return await super().translate_batch(images_with_configs, save_info)

    asyncio.run(run_agent_workflow(Preprocess(tmp_path), [(p, Cfg) for p in paths], {},
                                   client=ScriptedClient(FIRST_RUN * n_pages), log=lambda m: None,
                                   agent_settings={"inpaint_review": False, "recall_pass": False}))
    return folder, paths


def _json_path(folder, stem="001"):
    return folder / "manga_translator_work" / "json" / f"{stem}_translations.json"


def _read(folder, stem="001"):
    raw = json.loads(_json_path(folder, stem).read_text(encoding="utf-8"))
    return raw, next(iter(raw.values()))


def _write(folder, raw, stem="001"):
    _json_path(folder, stem).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")


def _store(folder):
    return AgentStore(str(folder / "manga_translator_work"))


def _user_edits(folder):
    """What the render and the user did after the run: vertical punctuation in #1 (the render), an
    editor field kept on #1, #2's text changed, #4 deleted, a box drawn at the bottom."""
    raw, page = _read(folder)
    regions = page["regions"]
    assert [r["translation"] for r in regions] == ["譯1……", "譯2", "譯4"]
    regions[0]["translation"] = "譯1︙"  # the render's vertical form; translation_raw keeps "譯1……"
    regions[0]["white_frame_rect_local"] = [-10, -40, 10, 40]
    regions[1]["translation"] = regions[1]["translation_raw"] = "我改的"
    del regions[2]  # #4: the user deleted the box
    regions.append({"lines": _quad(30, 190, 150, 220), "texts": ["note"], "text": "note", "translation": "我加的",
                    "font_size": 20, "direction": "h"})
    _write(folder, raw)


def test_text_key_ignores_what_the_render_and_line_breaks_change():
    assert text_key("……哦——……") == text_key("︙哦││︙")
    assert text_key("首先，暴雨一過[BR]草木。") == text_key("首先︐暴雨一過草木︒")
    assert text_key("（？）") == text_key("︵？︶")
    assert text_key("我改的") != text_key("譯2")


def test_a_page_comes_back_as_it_was_left_with_the_users_boxes_locked(tmp_path):
    folder, (path,) = _first_run(tmp_path)
    _user_edits(folder)
    doc = load_page(path, _store(folder))
    regions = [doc.regions[i] for i in doc.order]
    states = [(r.translation, r.skipped, r.user_edited, r.locked) for r in regions]
    assert states == [("譯1……", False, False, False),  # translation_raw: the agent's form, not the render's
                      ("我改的", False, True, True),
                      (None, True, False, False),  # #3 skipped by the run, back after #2 as it was
                      ("我加的", False, True, True)]  # drawn by the user
    skipped = next(r for r in regions if r.skipped)
    assert skipped.was_skipped and skipped.skip_category == "sign" and skipped.bbox[0] >= 195
    assert doc.removed and doc.removed[0]["text"] == "譯4"
    assert doc.missing() == []  # the page is done as it is
    mine = next(r for r in regions if r.translation == "我改的")
    for call in (lambda: doc.set_text(mine.id, "x", "新的"), lambda: doc.skip(mine.id, "other", "r"),
                 lambda: doc.set_line_breaks(mine.id, "我改[BR]的"), lambda: doc.merge([1, mine.id], "r")):
        with pytest.raises(RegionError, match="the user's"):
            call()
    # a region the agent leaves is written back as it was, the editor's fields with it
    out = doc.json_data()[doc.key]["regions"]
    _raw, before = _read(folder)
    assert out == before["regions"]


def test_the_lock_can_be_turned_off_and_the_agents_change_is_its_own(tmp_path):
    folder, (path,) = _first_run(tmp_path)
    _user_edits(folder)
    doc = load_page(path, _store(folder), lock=False)
    mine = next(r for r in doc.regions.values() if r.translation == "我改的")
    assert mine.user_edited and not mine.locked
    doc.set_text(mine.id, "text2", "改回來")
    assert not mine.user_edited
    first = doc.regions[doc.order[0]]
    doc.set_text(first.id, "text1", "新譯1")
    out = doc.json_data()[doc.key]["regions"]
    assert out[0]["translation"] == out[0]["translation_raw"] == "新譯1"
    assert "white_frame_rect_local" in out[0]  # the editor's own field stays; the measured box goes
    assert "render_box_rect_local" not in out[0]


def test_a_changed_source_text_or_white_frame_is_the_users_too(tmp_path):
    """Review item 01: the translation stays, only the OCR text of #1 or the white frame of #2 changed
    (the editor sets a custom frame when the frame is dragged or the font size changes)."""
    folder, (path,) = _first_run(tmp_path)
    side = _store(folder).load_sidecar("001")
    assert all("frame" in e["match"] and e["match"]["frame"] is None for e in side["regions"])
    raw, page = _read(folder)
    page["regions"][0]["text"] = page["regions"][0]["texts"][0] = "我改的原文"
    page["regions"][1].update(white_frame_rect_local=[-30, -60, 30, 60], has_custom_white_frame=True)
    _write(folder, raw)
    doc = load_page(path, _store(folder))
    first, second, third = (doc.regions[i] for i in doc.order if not doc.regions[i].skipped)
    assert (first.locked, second.locked, third.locked) == (True, True, False)
    for region in (first, second):
        for call in (lambda: doc.set_text(region.id, "x", "新的"), lambda: doc.skip(region.id, "other", "r"),
                     lambda: doc.split(region.id, region.bbox, "r")):
            with pytest.raises(RegionError, match="the user's"):
                call()
    # the next check: the frame the run left is not a change; moving it again is
    doc.status = "done"
    _store(folder).save_sidecar(doc.stem, doc.sidecar())
    side = _store(folder).load_sidecar("001")
    frames = [e["match"]["frame"] for e in side["regions"] if not e.get("skipped")]
    assert frames == [None, [-30.0, -60.0, 30.0, 60.0], None]
    for entry in side["regions"]:
        entry.pop("user_edited", None)
    _store(folder).save_sidecar(doc.stem, side)
    assert not load_page(path, _store(folder)).regions[second.id].locked
    page["regions"][1]["white_frame_rect_local"] = [-30, -50, 30, 70]
    _write(folder, raw)
    assert load_page(path, _store(folder)).regions[second.id].locked
    # a sidecar from before frames were kept: a custom frame there is the user's
    for entry in side["regions"]:
        entry["match"].pop("frame", None)
    _store(folder).save_sidecar(doc.stem, side)
    page["regions"][1]["white_frame_rect_local"] = [-30, -60, 30, 60]
    _write(folder, raw)
    assert load_page(path, _store(folder)).regions[second.id].locked


def test_deleted_boxes_stay_deleted(tmp_path):
    folder, (path,) = _first_run(tmp_path)
    _user_edits(folder)
    doc = load_page(path, _store(folder))
    x1, y1, x2, y2 = doc.removed[0]["bbox"]
    assert doc.overlap((x1 + 2, y1 + 2, x2 - 2, y2 - 2)) > 0.9  # the recall check counts it as boxed
    from manga_translator.agent.snap import SnapResult

    snap = SnapResult(bbox=(x1, y1, x2, y2), lines=_quad(x1, y1, x2, y2), font_size=20, fg=[0, 0, 0],
                      bg=[255, 255, 255], erase=np.ones((y2 - y1, x2 - x1), np.uint8), erase_origin=(x1, y1), strokes=4)
    with pytest.raises(RegionError, match="deleted the box here"):
        doc.add_region(snap, "v")
    # the next check still knows: the sidecar keeps it
    doc.status = "done"
    _store(folder).save_sidecar(doc.stem, doc.sidecar())
    assert load_page(path, _store(folder)).removed[0]["bbox"] == [x1, y1, x2, y2]


def test_the_saved_mask_stays_where_the_user_left_it(tmp_path):
    folder, (path,) = _first_run(tmp_path)
    raw, page = _read(folder)
    mask = _decode(page["mask_raw"])
    assert not mask[155:175, 205:275].any()  # the run took the skipped sign out of the mask
    mask[160:170, 210:270] = 255  # the user erased part of it in the editor anyway
    page["mask_raw"] = _mask_b64(mask)
    _write(folder, raw)
    doc = load_page(path, _store(folder))
    assert doc.output_mask() is None  # nothing changed: the user's mask stays
    sign = next(r for r in doc.regions.values() if r.skipped)
    # translated now: its strokes are found again and erased
    from manga_translator.agent.tools import AgentSession, ToolRegistry
    from manga_translator.agent.settings import AgentSettings

    session = AgentSession(AgentSettings(), _store(folder).memory(), [path])
    session.start_page(0, doc)
    result = ToolRegistry(session).execute("set_text", json.dumps({"id": sign.id, "source_text": "看板",
                                                                   "translation": "招牌", "category": "sign",
                                                                   "confidence": 0.9, "reason": "要翻"}))
    assert result.ok, result.text
    assert sign.erase is not None
    new = _decode(doc.output_mask())
    assert new[155:175, 205:275].sum() > mask[155:175, 205:275].sum()


class FakeFonts:
    default_font = "UserFont"

    @staticmethod
    def stand_in(value):
        return {"MingLiU": "STZhongsong", "Yu Mincho::Light": "STZhongsong", "DFKai-SB": ""}.get(value)

    @staticmethod
    def _same(a, b):
        return a.casefold() == b.casefold()


def test_thin_fonts_of_the_agent_move_and_the_users_fonts_stay(tmp_path):
    folder, (path,) = _first_run(tmp_path)
    raw, page = _read(folder)
    regions = page["regions"]
    rich = {"format": "richtext.v1", "blocks": [{"type": "paragraph", "inlines": [
        {"type": "text", "text": "譯", "style": {}}, {"type": "text", "text": "︙", "style": {"font_family": "Fallback"}}]}]}
    regions[0].update(font_family="Yu Mincho::Light", translation_rich=rich)  # the old font check's choice
    regions[1]["font_family"] = "DFKai-SB"  # the user's own choice in the editor
    regions[2]["font_family"] = "MingLiU"  # the agent's style
    _write(folder, raw)
    store = _store(folder)
    side = store.load_sidecar("001")
    side["fonts"] = {"changes": [{"region": 1, "from": "MingLiU", "to": "Yu Mincho::Light", "reason": "missing"}],
                     "borrowed": [{"region": 1, "font": "Yu Mincho::Light", "fallback": "Fallback"}]}
    for entry in side["regions"]:
        if entry.get("translation") in ("譯1……", "譯4"):
            entry["font"] = "MingLiU"
    store.save_sidecar("001", side)
    doc = load_page(path, store, fonts=FakeFonts())
    first, second, fourth = (doc.regions[i] for i in doc.order if not doc.regions[i].skipped)
    assert (first.font, first.user_font, first.drop_rich) == ("STZhongsong", False, True)
    assert (second.font, second.user_font) == ("", True)
    assert (fourth.font, fourth.user_font) == ("STZhongsong", False)
    out = doc.json_data()[doc.key]["regions"]
    assert [r["font_family"] for r in out] == ["STZhongsong", "DFKai-SB", "STZhongsong"]
    assert "translation_rich" not in out[0]  # the font check borrows again for the new font
    # the font check and the proofreader find the user's regions by center
    doc.status = "done"
    store.save_sidecar("001", doc.sidecar())
    assert json_center(out[1]) in locked_centers(store.load_sidecar("001"), "user_font")


def test_a_locked_region_gets_a_stand_in_for_the_agents_thin_font(tmp_path):
    # the user changed the text, not the agent's thin font (asked 2026-11-22)
    folder, (path,) = _first_run(tmp_path)
    _user_edits(folder)
    raw, page = _read(folder)
    page["regions"][1]["font_family"] = "MingLiU"  # the agent's thin font on the region the user changed
    _write(folder, raw)
    store = _store(folder)
    side = store.load_sidecar("001")
    next(e for e in side["regions"] if e.get("translation") == "譯2")["font"] = "MingLiU"
    store.save_sidecar("001", side)
    doc = load_page(path, store, fonts=FakeFonts())
    mine = next(r for r in doc.regions.values() if r.translation == "我改的")
    assert mine.locked and not mine.user_font and mine.font == "STZhongsong"
    out = doc.json_data()[doc.key]["regions"]
    assert out[1]["font_family"] == "STZhongsong" and out[1]["translation"] == "我改的"
    doc.status = "done"
    store.save_sidecar("001", doc.sidecar())
    side = store.load_sidecar("001")
    assert json_center(out[1]) in locked_centers(side, "locked")  # still the user's text, for the proofreader
    assert json_center(out[1]) not in locked_centers(side, "user_font")  # the font check measures it


def test_a_kai_face_moves_to_a_song_face_not_to_clerical_lettering():
    def face(family):
        return FontFace(path=f"{family}.ttf", index=0, family=family, style="Regular", weight=400, names=(family,),
                        locales=frozenset(), value=family)

    faces = {name: face(name) for name in ("DFKai-SB", "LiSu", "STZhongsong", "SimHei", "UserFont")}

    class Catalog:
        @staticmethod
        def find(value):
            return faces.get(value)

        @staticmethod
        def thin(f, kind):
            return f is not None and f.family == "DFKai-SB"

        @staticmethod
        def similarity(a, b, kind):  # LiSu's strokes look most like it
            return {"LiSu": 0.9, "STZhongsong": 0.6, "SimHei": 0.65}.get(b.family)

    service = FontService(Catalog(), None, "UserFont", "CHT")
    service.choices = [faces["LiSu"], faces["STZhongsong"], faces["SimHei"]]
    assert service.stand_in("DFKai-SB") == "STZhongsong"


def test_the_proofreader_leaves_locked_regions(tmp_path):
    folder, (path,) = _first_run(tmp_path)
    _user_edits(folder)
    store = _store(folder)
    doc = load_page(path, store)
    doc.status = "done"
    store.save_sidecar(doc.stem, doc.sidecar())
    page = ProofPage(path)
    page.match_sidecar(store.load_sidecar("001"))
    listing = page.listing()
    assert any("LOCKED" in line and "我改的" in line for line in listing)
    assert not any("LOCKED" in line and "譯1" in line for line in listing)


class PaintBackend:
    """Inpaints by painting the mask green."""

    def __init__(self):
        self.masks = []
        self.saved = {}

    def methods(self):
        return ["fake"]

    async def inpaint(self, method, image, mask):
        self.masks.append(mask.copy())
        out = np.array(image)
        out[mask > 0] = (0, 255, 0)
        return out

    def save(self, image_path, image):
        self.saved[image_path] = np.array(image)
        return image_path


def test_only_the_changed_areas_are_inpainted_again(tmp_path):
    image_path, json_path = make_page(tmp_path)
    source = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    before = tmp_path / "before.json"
    raw = json.loads(open(json_path, encoding="utf-8").read())
    page = next(iter(raw.values()))
    old = np.zeros((240, 320), np.uint8)
    old[20:60, 20:60] = 255  # erased before, kept now
    old[100:140, 100:140] = 255  # erased both times
    page["mask_raw"] = _mask_b64(old)
    before.write_text(json.dumps(raw), encoding="utf-8")
    new = np.zeros((240, 320), np.uint8)
    new[100:140, 100:140] = 255
    new[150:180, 200:280] = 255  # erased now
    page["mask_raw"] = _mask_b64(new)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh)
    inpainted = np.full((240, 320, 3), 128, np.uint8)  # the user's repaired image
    target = get_inpainted_path(image_path, create_dir=True)
    cv2.imwrite(target, inpainted)
    backend = PaintBackend()
    result = asyncio.run(inpaint_changes(image_path, str(before), backend, backend.save))
    assert result["status"] == "patched" and result["added"] == 30 * 80 and result["restored"] == 40 * 40
    out = backend.saved[image_path]
    assert (out[150:180, 200:280] == (0, 255, 0)).all()  # inpainted
    assert (out[20:60, 20:60] == source[20:60, 20:60]).all()  # the original back
    assert (out[100:140, 100:140] == 128).all() and (out[0:10, 0:10] == 128).all()  # the rest stays
    grown = backend.masks[0] > 0
    assert grown[150:180, 200:280].all() and not grown[100:140, 100:140].any()
    # nothing changed: nothing to do
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(json.loads(before.read_text(encoding="utf-8")), fh)
    assert asyncio.run(inpaint_changes(image_path, str(before), backend, backend.save))["status"] == "unchanged"


def test_a_check_run_keeps_the_page_and_reuses_the_inpainted_image(tmp_path):
    folder, paths = _first_run(tmp_path)
    _user_edits(folder)
    path = paths[0]
    cv2.imwrite(get_inpainted_path(path, create_dir=True), np.full((240, 320, 3), 128, np.uint8))
    user_json = _json_path(folder).read_bytes()
    doc = load_page(path, _store(folder))
    first, mine = doc.order[0], next(r.id for r in doc.regions.values() if r.translation == "我改的")
    script = [_calls(_say(mine, "不行"), _say(first, "新譯1")), _calls(("mark_page_done", {"confidence": 0.9}))]
    client = ScriptedClient(script)
    translator = FakeTranslator(tmp_path)
    logs = []
    asyncio.run(run_agent_workflow(translator, [(path, Cfg)], {}, client=client, log=logs.append, revise=True,
                                   inpaint_backend=PaintBackend(),
                                   agent_settings={"inpaint_review": True, "recall_pass": False}))
    # no preprocess; the render reuses the inpainted image (the mask did not change)
    assert translator.calls == [{"template": False, "save_text": True, "load_text": True, "skip_txt_import": True,
                                 "ignore_existing_inpainted": False}]
    page_text = client.requests[0][-1]["content"][0]["text"]
    assert "checked again" in page_text and "LOCKED" in page_text and "「我改的」" in page_text
    assert "Checking again" in client.requests[0][0]["content"]
    tool_results = [m for m in client.requests[1] if m.get("role") == "tool"]
    assert "the user's" in tool_results[0]["content"]
    _raw, page = _read(folder)
    assert [r["translation"] for r in page["regions"]] == ["新譯1", "我改的", "我加的"]
    side = _store(folder).load_sidecar("001")
    assert [bool(e.get("locked")) for e in side["regions"] if not e.get("skipped")] == [False, True, True]
    backups = os.listdir(folder / "manga_translator_work" / "agent" / "backups")
    assert any(n.startswith("001_translations.") for n in backups) and any(n.endswith(".png") for n in backups)
    copy = next(n for n in backups if n.startswith("001_translations."))
    assert (folder / "manga_translator_work" / "agent" / "backups" / copy).read_bytes() == user_json
    assert not (folder / "manga_translator_work" / "agent" / "run_journal.json").exists()


# --- the box set in the editor ---------------------------------------------------------
def test_text_goes_in_the_box_set_in_the_editor():
    """ch34 p6: a label the user moved and spaced in the editor was typeset around the detected lines."""
    from manga_translator.agent.workflow import editor_box_center

    label = {"lines": [[[335, 579], [676, 579], [676, 740], [335, 740]]], "center": [505.5, 659.5],
             "render_box_rect_local": [-141, -35, 9, 87]}
    editor_box_center(label)
    assert label["center"] == pytest.approx([439.5, 685.5])
    # the user's own white frame comes first; older files kept the render box as the white frame
    framed = {"center": [100, 100], "render_box_rect_local": [-10, -10, 10, 10],
              "white_frame_rect_local": [0, 0, 40, 20], "has_custom_white_frame": True}
    older = {"center": [100, 100], "white_frame_rect_local": [0, 0, 40, 20]}
    for region in (framed, older):
        editor_box_center(region)
        assert region["center"] == pytest.approx([120, 110])
    # turned with the region; without a center, from the middle of the lines like the editor
    turned = {"center": [100, 100], "angle": 90, "render_box_rect_local": [10, -5, 30, 5]}
    editor_box_center(turned)
    assert turned["center"] == pytest.approx([100, 120])
    bare = {"lines": [[[0, 0], [40, 0], [40, 20], [0, 20]]], "render_box_rect_local": [0, 0, 10, 10]}
    editor_box_center(bare)
    assert bare["center"] == pytest.approx([25, 15])
    # no box (a new region, or one the agent changed): as it was
    for region in ({"center": [5, 5]}, {"center": [5, 5], "render_box_rect_local": [0, "x", 1, 1]}):
        before = dict(region)
        editor_box_center(region)
        assert region == before


def test_the_render_reads_each_region_through_the_hook(tmp_path):
    from manga_translator.agent.workflow import editor_box_center
    from manga_translator.config import Config, TranslatorConfig
    from manga_translator.manga_translator import MangaTranslator

    image_path, json_path = make_page(tmp_path)
    raw = json.load(open(json_path, encoding="utf-8"))
    next(iter(raw.values()))["regions"][0].update(translation="譯", center=[40, 70],
                                                   render_box_rect_local=[10, 20, 30, 40])
    json.dump(raw, open(json_path, "w", encoding="utf-8"), ensure_ascii=False)
    saved = open(json_path, encoding="utf-8").read()
    translator = MangaTranslator(params={"load_text": True, "translator": "none"})
    cfg = Config(translator=TranslatorConfig(translator="none"))

    def first_center():
        regions, *_rest, failures = translator._load_text_and_regions_from_file(image_path, cfg)
        assert len(regions) == 3 and failures == 0
        return [float(v) for v in regions[0].center]

    assert first_center() == [40, 70]  # off by default: other modes read the regions as saved
    translator.load_text_region_hook = editor_box_center
    assert first_center() == [60, 100]
    translator.load_text_region_hook = lambda region: 1 / 0
    assert first_center() == [40, 70]  # a failing hook loses no region
    assert open(json_path, encoding="utf-8").read() == saved  # the JSON is only read
