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


def test_the_editor_never_exports_over_the_source_image(tmp_path):
    """The project review of 2026-11-22 (P1-02): a new setup, no output folder, the format kept and
    overwrite off; the export replaced the source PNG."""
    from services.export_service import ExportService

    chapter, image = _chapter(tmp_path)
    result = chapter / "manga_translator_work" / "result" / "001.png"
    service = ExportService()
    assert service.build_output_path({"cli": {"format": "", "overwrite": False}, "app": {}}, str(image)) == \
        str(result)
    # an output folder that was removed
    gone = {"cli": {"format": ""}, "app": {"last_output_path": str(tmp_path / "removed")}}
    assert service.build_output_path(gone, str(image)) == str(result)
    # the output folder is the source folder: the same name and format would be the source
    same = {"cli": {"format": ""}, "app": {"last_output_path": str(chapter)}}
    assert service.build_output_path(same, str(image)) == str(result)
    assert service.build_output_path({**same, "cli": {"format": "jpg"}}, str(image)) == str(chapter / "001.jpg")
    _write_json(image, last_export_dir=str(chapter))  # a saved folder that is the source folder
    assert service.build_output_path({"cli": {"format": ""}, "app": {}}, str(image)) == str(result)
    assert service.is_source_file(str(chapter / "." / "001.png"), str(image))


def test_an_export_job_over_its_source_is_refused(tmp_path):
    from services.export_service import ExportService

    chapter, image = _chapter(tmp_path)
    before = image.read_bytes()
    job = type("Job", (), {"source_path": str(image), "output_path": str(image)})()
    result = ExportService().execute_export_job(job)
    assert result.error and "source image" in result.error
    assert image.read_bytes() == before


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


def test_pages_that_share_a_stem_get_output_images_of_their_own(tmp_path):
    """Review 2026-11-24 B2: ``001.jpg`` and ``001.png`` of one folder saved as PNG were both written
    to ``001.png`` by the batch, the desktop precheck and the editor (the later one replaced the
    earlier, or was skipped as already done); the CLI took them as one output."""
    from app_logic import TranslationWorker
    from manga_translator.mode.subprocess_manager import same_output_paths
    from manga_translator.utils.path_manager import output_file_name

    chapter, png = _chapter(tmp_path)
    jpg = chapter / "001.jpg"
    Image.new("RGB", (64, 64), "red").save(jpg)
    other = chapter / "002.jpg"
    Image.new("RGB", (64, 64), "blue").save(other)
    assert [output_file_name(str(p), "png") for p in (jpg, png, other)] == ["001.jpg.png", "001.png", "002.png"]
    assert [output_file_name(str(p), fmt) for p, fmt in ((jpg, None), (png, "不指定"), (jpg, "jpg"))] == \
        ["001.jpg", "001.png", "001.jpg"]  # its own format: its own name

    output = tmp_path / "output"
    save_info = {"output_folder": str(output), "format": "png", "input_folders": {os.path.normpath(str(chapter))}}
    batch = [MangaTranslator._calculate_output_path(None, str(p), save_info) for p in (jpg, png)]
    assert batch == [str(output / "ch35" / "001.jpg.png"), str(output / "ch35" / "001.png")]
    logic = type("Logic", (), {"file_to_folder_map": {}})()
    precheck = [TranslationWorker._calculate_output_path(logic, str(p), save_info) for p in (jpg, png)]
    assert [os.path.basename(p) for p in precheck] == ["001.jpg.png", "001.png"]
    editor = [_export_path(tmp_path, p, format="png") for p in (jpg, png)]
    assert [os.path.basename(p) for p in editor] == ["001.jpg.png", "001.png"]
    assert same_output_paths([str(png), str(jpg)], [], "png") == {}

    os.remove(png)  # 001.jpg is the only 001 again: the name it always had
    assert output_file_name(str(jpg), "png") == "001.png"
