"""Review 2026-11-28 B8: a turned text's lines are saved upright about its center with the turn in
``angle``; the sidecar keeps the page's (turned) lines. The font check compared the JSON's upright
center with the sidecar's, the stages that turn the lines back matched centers exactly (the float
error of the two turns moved a box's floor or ceil by a pixel), and a skipped turned box restored in
the editor came back turned twice."""
import copy
import math

from manga_translator.agent import review
from manga_translator.agent.page import Centers, Region
from manga_translator.agent.revise import json_center, locked_centers
from manga_translator.utils.textblock import page_lines_to_saved, saved_lines_to_page


def _turned(angle=30.0, center=(150.0, 120.0), w=40.0, h=160.0):
    """A JSON region as saved: one upright line about ``center`` and the turn in ``angle``."""
    cx, cy = center
    quad = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]
    return {"lines": [quad], "angle": angle, "center": [cx, cy], "texts": ["x"], "translation": "譯"}


def test_centers_are_found_within_two_pixels():
    found = Centers([((10.0, 10.0), "a"), ((10.5, 10.0), "b"), ((40.0, 40.0), "c")])
    assert found.get((10.5, 10.0)) == "b" and found.get((10.4, 10.1)) == "b"  # exact, then the nearest
    assert found.get((41.5, 38.5)) == "c" and found.get((42.5, 40.0)) is None
    assert (39.5, 40.5) in found and (100, 100) not in found and bool(found) and not Centers()
    assert Centers([((1, 1), "old"), ((1, 1), "new")]).get((1, 1)) == "new"  # as a dict


def test_the_font_check_finds_the_users_font_on_a_turned_box():
    saved = [_turned(angle) for angle in (5.0, 17.0, 33.0, 48.0, 61.0, 79.0)]
    for region in saved:
        on_page = Region(id=1, data=saved_lines_to_page(copy.deepcopy(region)))  # as the agent's sidecar has it
        side = {"regions": [{"id": 1, "user_font": True, "match": {"center": on_page.center, "bbox": on_page.bbox}}]}
        assert json_center(region) in locked_centers(side, "user_font"), region["angle"]
        # once more through the JSON: saved upright by the agent, read back by the proofreader
        again = Region(id=1, data=saved_lines_to_page(page_lines_to_saved(copy.deepcopy(on_page.data))))
        assert again.center in locked_centers(side, "user_font")


def test_a_skipped_turned_box_is_restored_upright_and_found_where_it_was():
    region = Region(id=4, data=saved_lines_to_page(_turned(30.0)))  # the sidecar's "json": lines on the page
    item = review.Skipped(id=4, category="sfx", text="ドン", reason="", center=region.center, bbox=region.bbox,
                          data=copy.deepcopy(region.data))
    data = review.restore_data(item, [])
    assert data["angle"] == 30.0 and data["center"] == [150.0, 120.0]
    upright = data["lines"][0]
    assert math.isclose(upright[0][1], upright[1][1], abs_tol=1e-6)  # the top edge is level: saved upright
    box = review.region_bbox(data, upright=True)
    assert all(abs(a - b) <= 1 for a, b in zip(box, region.bbox))
    assert review.match_region(None, region.bbox, [data]) == 0  # the editor sees it back: no second restore
