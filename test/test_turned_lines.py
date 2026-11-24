"""A turned text (``angle``) is saved upright about its center (``TextBlock.to_dict``, the editor) and
taken back onto the page when it is loaded (the render, the agent, the proofreader). Loaded as saved, a
label written on a 38° slant, 178 x 24 px, came back as a 156 x 128 box and was drawn at 64 px instead
of 15 (ch42 p1, the road labels of the map)."""
import json

import cv2
import numpy as np
import pytest

from manga_translator.agent.page import PageDoc
from manga_translator.agent.proofread import ProofPage
from manga_translator.utils import TextBlock
from manga_translator.utils.textblock import page_lines_to_saved, rotate_polygons, saved_lines_to_page

CENTER = (160.0, 120.0)
UPRIGHT = [[[71.0, 108.0], [249.0, 108.0], [249.0, 132.0], [71.0, 132.0]]]
"""A 178 x 24 line about CENTER."""
ANGLE = -38.0


def _on_page():
    """UPRIGHT turned onto the page: how the detector and the agent's snap give a slanted line."""
    flat = np.asarray(UPRIGHT, dtype=np.float64).reshape(-1, 8)
    return rotate_polygons(np.asarray(CENTER), flat, -ANGLE, to_int=False).reshape(-1, 4, 2).astype(np.float64)


def _block(lines, center=None):
    return TextBlock(lines=np.asarray(lines, dtype=np.float64), texts=["バリケード"], angle=ANGLE, font_size=17,
                     direction="h", translation="路障", center=center)


def _saved_region():
    return {"lines": UPRIGHT, "center": list(CENTER), "angle": ANGLE, "texts": ["バリケード"], "text": "バリケード",
            "translation": "路障", "font_size": 17, "direction": "h", "prob": 1.0}


def _write_page(tmp_path, regions):
    folder = tmp_path / "chapter"
    json_dir = folder / "manga_translator_work" / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    image_path = str(folder / "001.png")
    cv2.imwrite(image_path, np.full((240, 320, 3), 255, np.uint8))
    data = {image_path: {"regions": regions, "skip_font_scaling": False, "original_width": 320,
                         "original_height": 240}}
    json_path = json_dir / "001_translations.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    return image_path, str(json_path)


def test_the_lines_turn_back_and_forth():
    region = {"lines": _on_page().tolist(), "center": list(CENTER), "angle": ANGLE}
    saved = page_lines_to_saved(json.loads(json.dumps(region)))
    assert np.allclose(saved["lines"], UPRIGHT, atol=1e-6)
    assert np.allclose(saved_lines_to_page(saved)["lines"], _on_page(), atol=1e-6)
    # not turned: left as it is
    flat = {"lines": UPRIGHT, "angle": 0}
    assert saved_lines_to_page(dict(flat))["lines"] is UPRIGHT


def test_a_saved_text_comes_back_the_size_it_was():
    block = _block(_on_page())
    width, height = block.unrotated_size
    assert width == pytest.approx(178, abs=2) and height == pytest.approx(24, abs=2)
    saved = block.to_dict()
    assert np.ptp(np.asarray(saved["lines"])[..., 1]) == pytest.approx(24, abs=1)  # upright, as the editor draws it
    loaded = saved_lines_to_page(dict(saved))
    again = _block(loaded["lines"], loaded["center"])
    assert again.unrotated_size == pytest.approx((width, height), abs=2)  # TextBlock rounds to pixels
    # taken as it was saved, the line was turned a second time
    assert _block(saved["lines"], saved["center"]).unrotated_size[1] > 100


def test_the_render_loads_a_saved_turned_text_on_the_page(tmp_path):
    from manga_translator.config import Config, TranslatorConfig
    from manga_translator.manga_translator import MangaTranslator

    image_path, _ = _write_page(tmp_path, [_saved_region()])
    translator = MangaTranslator(params={"load_text": True, "translator": "none"})
    regions, *_ = translator._load_text_and_regions_from_file(image_path,
                                                              Config(translator=TranslatorConfig(translator="none")))
    assert len(regions) == 1
    assert regions[0].unrotated_size == pytest.approx((178, 24), abs=2)
    assert np.allclose(regions[0].lines, _on_page(), atol=1e-4)


def test_the_agent_reads_and_writes_the_saved_form(tmp_path):
    image_path, json_path = _write_page(tmp_path, [_saved_region()])
    doc = PageDoc.load(image_path, json_path)
    region = doc.regions[1]
    assert np.allclose(region.polygons[0], _on_page()[0], atol=1e-4)  # where the text lies on the page
    region.translation = "路障"
    out = doc.to_json_regions()
    assert out[0]["angle"] == ANGLE
    assert np.allclose(out[0]["lines"], UPRIGHT, atol=0.6)  # the page lines about the new center
    # the render takes it back to the page: the same lines the agent had
    assert np.allclose(saved_lines_to_page(out[0])["lines"], region.polygons[0][None], atol=1e-4)


def test_the_proofreader_sees_the_text_where_the_agent_saw_it(tmp_path):
    image_path, json_path = _write_page(tmp_path, [_saved_region()])
    doc = PageDoc.load(image_path, json_path)
    proof = ProofPage(image_path)
    assert proof.regions[1].bbox == doc.regions[1].bbox
    assert proof.regions[1].center == doc.regions[1].center


def test_a_check_finds_the_agents_turned_box_after_the_render(tmp_path):
    """The render saves the agent's slanted label upright about its center; the check takes it back onto
    the page, so it is the agent's box where the sidecar left it, not one the user moved (and locked)."""
    from manga_translator.agent.revise import load_page
    from manga_translator.agent.store import AgentStore
    from test_agent_revise import _first_run, _read, _write

    folder, (path,) = _first_run(tmp_path)
    raw, page = _read(folder)
    region = page["regions"][0]
    cx, cy = region["center"]
    upright = [[[cx - 50, cy - 10], [cx + 50, cy - 10], [cx + 50, cy + 10], [cx - 50, cy + 10]]]
    region.update(lines=upright, angle=-30.0)  # as TextBlock.to_dict saves a line on a 30° slant
    _write(folder, raw)
    on_page = saved_lines_to_page(dict(region))["lines"]
    xs, ys = [p[0] for p in on_page[0]], [p[1] for p in on_page[0]]
    store = AgentStore(str(folder / "manga_translator_work"))
    side = store.load_sidecar("001")
    entry = next(e for e in side["regions"] if e.get("translation") == "譯1……")
    entry["match"]["bbox"] = [min(xs), min(ys), max(xs), max(ys)]  # the agent's box: the line on the page
    store.save_sidecar("001", side)

    doc = load_page(path, store)
    first = doc.regions[doc.order[0]]
    assert first.translation == "譯1……"
    assert not first.user_edited and not first.locked
    assert np.allclose(first.polygons[0], on_page[0], atol=1e-4)
    # left as it was: written back the way the render saved it
    out = doc.json_data()[doc.key]["regions"][0]
    assert np.allclose(out["lines"], upright, atol=1e-6) and out["angle"] == -30.0
