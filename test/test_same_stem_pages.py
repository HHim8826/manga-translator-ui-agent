"""``001.jpg`` and ``001.png`` in one folder keep their own work files (the project review of 2026-11-22,
P1-03: both were saved to ``001_translations.json`` and both opened as the last one saved)."""
import _bootstrap  # noqa: F401, I001

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from manga_translator.agent.store import page_stem
from manga_translator.utils.path_manager import (
    find_json_path,
    get_json_path,
    get_original_txt_path,
    json_image_candidates,
    page_name,
)


def _image(path, color):
    Image.new("RGB", (40, 40), color).save(path)
    return str(path)


def _region(text):
    points = [[10.0, 10.0], [30.0, 10.0], [30.0, 20.0], [10.0, 20.0]]
    return {"text": "原文", "translation": text, "center": [20.0, 15.0], "lines": [points], "polygons": [points]}


def _load(image_path):
    from services.file_service import FileService

    with patch("services.get_config_service", return_value=SimpleNamespace(get_config=lambda: None)):
        regions, _mask, _size, _overlays = FileService().load_translation_json(image_path)
    return [r["translation"] for r in regions]


def test_pages_that_share_a_stem_get_work_files_of_their_own(tmp_path):
    jpg = _image(tmp_path / "001.jpg", "red")
    png = _image(tmp_path / "001.png", "blue")
    other = _image(tmp_path / "002.png", "green")
    assert (page_name(jpg), page_name(png), page_name(other)) == ("001.jpg", "001.png", "002")
    assert page_stem(png) == "001.png"
    assert os.path.basename(get_json_path(jpg, create_dir=False)) == "001.jpg_translations.json"
    assert os.path.basename(get_json_path(other, create_dir=False)) == "002_translations.json"  # as before
    assert os.path.basename(get_original_txt_path(png, create_dir=False, output_format="txt")) == \
        "001.png_original.txt"
    os.remove(jpg)  # the folder changed: 001.png is the only 001 again
    assert page_name(png) == "001"


def test_two_pages_that_share_a_stem_keep_their_own_translations(tmp_path):
    from services.export_service import ExportService

    jpg = _image(tmp_path / "001.jpg", "red")
    png = _image(tmp_path / "001.png", "blue")
    ExportService().save_editor_project(jpg, [_region("JPEG-edit")], None, None)
    ExportService().save_editor_project(png, [_region("PNG-edit")], None, None)
    assert find_json_path(jpg) != find_json_path(png)
    assert _load(png) == ["PNG-edit"] and _load(jpg) == ["JPEG-edit"]  # opened the other way round


def test_an_old_shared_json_stays_with_the_image_it_is_of(tmp_path):
    jpg = _image(tmp_path / "001.jpg", "red")
    legacy = tmp_path / "manga_translator_work" / "json" / "001_translations.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({os.path.abspath(jpg): {"regions": [_region("舊的")]}}, ensure_ascii=False),
                      encoding="utf-8")
    assert find_json_path(jpg) == str(legacy)  # one 001: the old name as before
    png = _image(tmp_path / "001.png", "blue")  # then another 001 is added
    assert find_json_path(jpg) == str(legacy) and find_json_path(png) is None
    assert _load(png) == [] and _load(jpg) == ["舊的"]
    # the text export of that JSON goes to the image it is of, not to the first extension found
    first = json_image_candidates(str(tmp_path), "001", str(legacy))
    assert os.path.basename(first[0]) == "001.jpg"
    # an old JSON whose entries do not say which image it is of is used by neither
    legacy.write_text(json.dumps({"001": {"regions": [_region("不知道是誰的")]}}, ensure_ascii=False),
                      encoding="utf-8")
    assert find_json_path(jpg) is None and find_json_path(png) is None


def test_the_reader_uses_another_images_entry_only_when_no_other_image_has_the_stem(tmp_path):
    png = _image(tmp_path / "001.png", "blue")
    json_path = get_json_path(png)
    moved = os.path.join(str(tmp_path / "elsewhere"), "001.jpg")  # converted from a JPEG somewhere else
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({moved: {"regions": [_region("轉檔前")]}}, fh, ensure_ascii=False)
    assert _load(png) == ["轉檔前"]  # as before: the only 001 of the folder


def test_a_page_keeps_its_own_json_when_the_other_001_is_moved_away(tmp_path):
    """Review 2026-11-26 P1-2: with 001.jpg renamed, 001.png went by ``001`` again and opened the
    JPEG's old ``001_translations.json`` (or nothing, when both were saved under their file names)."""
    from services.export_service import ExportService

    jpg = _image(tmp_path / "001.jpg", "red")
    ExportService().save_editor_project(jpg, [_region("FIRST-JPG")], None, None)  # alone: ``001``
    png = _image(tmp_path / "001.png", "blue")
    ExportService().save_editor_project(png, [_region("LATEST-PNG")], None, None)
    assert _load(png) == ["LATEST-PNG"] and _load(jpg) == ["FIRST-JPG"]
    os.rename(jpg, tmp_path / "001_old.jpg")
    assert page_name(png) == "001.png" and _load(png) == ["LATEST-PNG"]
    ExportService().save_editor_project(png, [_region("PNG-AGAIN")], None, None)
    assert os.path.basename(find_json_path(png)) == "001.png_translations.json"
    os.rename(tmp_path / "001_old.jpg", jpg)  # back again: neither took the other's JSON
    assert _load(jpg) == ["FIRST-JPG"] and _load(png) == ["PNG-AGAIN"]

    both = tmp_path / "both"
    both.mkdir()
    jpg, png = _image(both / "001.jpg", "red"), _image(both / "001.png", "blue")
    ExportService().save_editor_project(jpg, [_region("JPG")], None, None)
    ExportService().save_editor_project(png, [_region("PNG")], None, None)
    os.remove(jpg)
    assert _load(png) == ["PNG"]  # was empty: its own JSON was not looked for


def test_an_image_named_after_another_images_file_name_is_a_page_of_its_own(tmp_path):
    """Review 2026-11-26 P1-3: 001.jpg (beside 001.png) and 001.jpg.png both went by ``001.jpg``: one
    JSON for the two, and 001.jpg as PNG was exported to 001.jpg.png, the other page's output."""
    from services.export_service import ExportService
    from manga_translator.utils.path_manager import output_file_name

    pages = [_image(tmp_path / name, color) for name, color in
             (("001.jpg", "red"), ("001.png", "blue"), ("001.jpg.png", "green"))]
    assert len({get_json_path(page, create_dir=False) for page in pages}) == 3
    for page, text in reversed(list(zip(pages, ("JPG", "PNG", "JPG-PNG")))):
        ExportService().save_editor_project(page, [_region(text)], None, None)
    assert [_load(page) for page in reversed(pages)] == [["JPG-PNG"], ["PNG"], ["JPG"]]
    as_png = [output_file_name(page, "png") for page in pages]
    assert as_png == ["001_jpg.png", "001.png", "001.jpg.png"] and len(set(as_png)) == 3
    as_webp = [output_file_name(page, "webp") for page in pages]
    assert as_webp == ["001.jpg.webp", "001.png.webp", "001.jpg.png.webp"]
    plain = tmp_path / "plain"
    plain.mkdir()
    assert output_file_name(_image(plain / "002.jpg", "red"), "png") == "002.png"  # as before


def test_an_old_json_of_the_other_image_is_not_read_while_that_image_is_there(tmp_path):
    """An older version wrote 001.jpg.png's JSON as ``001.jpg``: 001.jpg (beside 001.png) does not
    open it, and 001.jpg.png still does."""
    jpg = _image(tmp_path / "001.jpg", "red")
    _image(tmp_path / "001.png", "blue")
    other = _image(tmp_path / "001.jpg.png", "green")
    old = tmp_path / "manga_translator_work" / "json" / "001.jpg_translations.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({os.path.abspath(other): {"regions": [_region("舊版")]}}, ensure_ascii=False),
                   encoding="utf-8")
    assert find_json_path(jpg) is None and _load(jpg) == []
    assert find_json_path(other) == str(old) and _load(other) == ["舊版"]


def test_the_agent_tells_the_two_pages_apart(app):  # noqa: F811
    from test_agents_page import _t
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([{"type": "run_start", "run": "r1", "batch": "b1", "chapter": "c", "pages": ["001.jpg", "001.png"],
                  "stems": ["001.jpg", "001.png"], "model": "m"}])
    assert len(page._order) == 2


from test_agents_page import app  # noqa: E402, F401  (one QApplication for the session)
