"""A page typeset from its JSON (how the agent workflow renders) remembers the chapter's output
folder, and the editor exports a page that has no such record to its chapter folder too, not to
the output root, where pages of different chapters with the same name replace each other."""
import _bootstrap  # noqa: F401, I001

import asyncio
import json
import os

from PIL import Image

from manga_translator.config import Config, Translator, TranslatorConfig
from manga_translator.manga_translator import MangaTranslator
from manga_translator.utils.path_manager import get_json_path


def _chapter(tmp_path, name="ch35"):
    chapter = tmp_path / "manga" / name
    chapter.mkdir(parents=True)
    image = chapter / "001.png"
    Image.new("RGB", (64, 64), "white").save(image)
    return chapter, image


def _write_json(image, **fields):
    path = get_json_path(str(image), create_dir=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(image): {"regions": [], **fields}}, f, ensure_ascii=False)
    return path


def _saved_dir(image):
    with open(get_json_path(str(image), create_dir=False), encoding="utf-8") as f:
        return next(iter(json.load(f).values())).get("last_export_dir")


def test_a_page_typeset_from_its_json_remembers_the_chapter_folder(tmp_path):
    chapter, image = _chapter(tmp_path)
    _write_json(image)
    output = tmp_path / "output"
    translator = MangaTranslator(params={"translator": "none", "use_gpu": False, "filter_text_enabled": False})
    translator.load_text = translator.save_text = translator.skip_txt_import = True
    save_info = {"output_folder": str(output), "format": None, "overwrite": True,
                 "input_folders": {os.path.normpath(str(chapter))}}
    config = Config(translator=TranslatorConfig(translator=Translator.none))
    asyncio.run(translator.translate_batch([(str(image), config)], save_info=save_info))

    assert (output / "ch35" / "001.png").exists()
    assert _saved_dir(image) == os.path.normpath(str(output / "ch35"))


def _export_path(tmp_path, image, **cli):
    from services.export_service import ExportService

    config = {"cli": {"format": "", **cli}, "app": {"last_output_path": str(tmp_path / "output")}}
    return ExportService().build_output_path(config, str(image))


def _map(folder, image):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "translation_map.json", "w", encoding="utf-8") as f:
        json.dump({os.path.normpath(str(folder / image.name)): os.path.normpath(str(image))}, f)


def test_the_editor_exports_a_page_without_a_record_to_its_chapter_folder(tmp_path):
    chapter, image = _chapter(tmp_path)
    _write_json(image)  # typeset before the fix: no last_export_dir
    other, other_image = _chapter(tmp_path, "ch34")
    _map(tmp_path / "output" / "ch34", other_image)
    _map(tmp_path / "output" / "ch35", image)

    assert _export_path(tmp_path, image) == str(tmp_path / "output" / "ch35" / "001.png")
    assert _export_path(tmp_path, other_image) == str(tmp_path / "output" / "ch34" / "001.png")


def test_the_editor_keeps_its_other_choices(tmp_path):
    chapter, image = _chapter(tmp_path)
    _write_json(image)
    (tmp_path / "output").mkdir()
    # never typeset into the output folder: the output root, as before
    assert _export_path(tmp_path, image) == str(tmp_path / "output" / "001.png")
    # a folder the page was saved to before still wins over the map
    _map(tmp_path / "output" / "ch35", image)
    saved = tmp_path / "elsewhere"
    _write_json(image, last_export_dir=str(saved))
    assert _export_path(tmp_path, image) == str(saved / "001.png")
