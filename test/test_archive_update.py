"""An archive imported again after it changed keeps the work of its pages (the project review of
2026-11-22, P1-01: the extraction folder, with its manga_translator_work, was emptied and extracted
again, and the proofread JSON and the AI Agent's backups were gone)."""
import io
import json
import os
import time
import zipfile

from PIL import Image

from desktop_qt_ui.utils.archive_extractor import extract_images_from_archive


def _png(color):
    data = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(data, format="PNG")
    return data.getvalue()


def _cbz(path, pages):
    with zipfile.ZipFile(path, "w") as zf:
        for name, color in pages:
            zf.writestr(name, _png(color))
    stamp = time.time() + len(pages) * 10  # a newer version: its time differs
    os.utime(path, (stamp, stamp))


def _work(folder):
    return os.path.join(folder, "manga_translator_work")


def _proofread(folder, image_name, text):
    """What the user's work leaves for a page: its JSON, a repaired inpainted image, an agent sidecar."""
    stem = os.path.splitext(image_name)[0]
    image = os.path.join(folder, image_name)
    for sub, name, data in (("json", f"{stem}_translations.json",
                             json.dumps({image: {"regions": [{"translation": text}]}}, ensure_ascii=False)),
                            ("inpainted", f"{stem}_inpainted.png", "my repairs " + text),
                            (os.path.join("agent", "pages"), f"{stem}.agent.json", json.dumps({"status": "done"}))):
        os.makedirs(os.path.join(_work(folder), sub), exist_ok=True)
        with open(os.path.join(_work(folder), sub, name), "w", encoding="utf-8") as fh:
            fh.write(data)


def _translation(folder, image_name):
    stem = os.path.splitext(image_name)[0]
    path = os.path.join(_work(folder), "json", f"{stem}_translations.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return next(iter(json.load(fh).values()))["regions"][0]["translation"]


def test_a_page_added_in_front_keeps_the_work_of_the_others(tmp_path):
    archive = str(tmp_path / "第1話.cbz")
    _cbz(archive, [("b.png", "red"), ("c.png", "green")])
    folder = str(tmp_path / "out" / "第1話" / "original_images")
    images, _ = extract_images_from_archive(archive, folder)
    assert [os.path.basename(p) for p in images] == ["0000_b.png", "0001_c.png"]
    _proofread(folder, "0000_b.png", "紅頁")
    _proofread(folder, "0001_c.png", "綠頁")
    backup = os.path.join(_work(folder), "agent", "backups", "0000_b_translations.20261122-101010.json")
    os.makedirs(os.path.dirname(backup))
    with open(backup, "w", encoding="utf-8") as fh:
        fh.write("{}")
    with open(os.path.join(_work(folder), "agent", "saved_json.json"), "w", encoding="utf-8") as fh:
        json.dump({"0000_b": "print-b", "0001_c": "print-c"}, fh)

    _cbz(archive, [("a.png", "blue"), ("b.png", "red"), ("c.png", "green")])  # a page in front
    images, _ = extract_images_from_archive(archive, folder)
    assert [os.path.basename(p) for p in images] == ["0000_a.png", "0001_b.png", "0002_c.png"]
    assert _translation(folder, "0000_a.png") is None  # the new page has no work yet
    assert _translation(folder, "0001_b.png") == "紅頁" and _translation(folder, "0002_c.png") == "綠頁"
    with open(os.path.join(_work(folder), "inpainted", "0001_b_inpainted.png"), encoding="utf-8") as fh:
        assert fh.read() == "my repairs 紅頁"
    assert os.path.exists(os.path.join(_work(folder), "agent", "pages", "0002_c.agent.json"))
    assert os.path.exists(backup)  # the AI Agent's backups stay
    with open(os.path.join(_work(folder), "agent", "saved_json.json"), encoding="utf-8") as fh:
        assert json.load(fh) == {"0001_b": "print-b", "0002_c": "print-c"}
    assert Image.open(images[1]).getpixel((0, 0)) == (255, 0, 0)
    assert not [n for n in os.listdir(os.path.dirname(folder)) if n.startswith(".extract-")]  # no leftovers
    # imported again unchanged: the cache, nothing moves
    assert extract_images_from_archive(archive, folder)[0] == images


def test_a_page_that_changed_or_went_is_kept_aside_with_its_work(tmp_path):
    archive = str(tmp_path / "第2話.cbz")
    _cbz(archive, [("a.png", "red"), ("b.png", "green")])
    folder = str(tmp_path / "out" / "第2話" / "original_images")
    extract_images_from_archive(archive, folder)
    _proofread(folder, "0000_a.png", "舊的紅頁")
    _proofread(folder, "0001_b.png", "綠頁")

    _cbz(archive, [("a.png", "blue")])  # a's picture changed, b is gone
    images, _ = extract_images_from_archive(archive, folder)
    assert [os.path.basename(p) for p in images] == ["0000_a.png"]
    assert _translation(folder, "0000_a.png") is None  # the old text is not put on the new picture
    replaced = os.path.join(_work(folder), "replaced")
    [aside] = os.listdir(replaced)
    aside = os.path.join(replaced, aside)
    with open(os.path.join(aside, "json", "0000_a_translations.json"), encoding="utf-8") as fh:
        assert "舊的紅頁" in fh.read()
    assert os.path.exists(os.path.join(aside, "json", "0001_b_translations.json"))
    assert os.path.exists(os.path.join(aside, "inpainted", "0001_b_inpainted.png"))
    assert Image.open(os.path.join(aside, "original_images", "0000_a.png")).getpixel((0, 0)) == (255, 0, 0)
    assert Image.open(images[0]).getpixel((0, 0)) == (0, 0, 255)
    assert not os.path.exists(os.path.join(folder, "0001_b.png"))


def test_a_folder_without_work_files_is_extracted_again_as_before(tmp_path):
    archive = str(tmp_path / "第3話.cbz")
    _cbz(archive, [("a.png", "red")])
    folder = str(tmp_path / "out" / "第3話" / "original_images")
    extract_images_from_archive(archive, folder)
    _cbz(archive, [("a.png", "red"), ("b.png", "green")])
    images, _ = extract_images_from_archive(archive, folder)
    assert [os.path.basename(p) for p in images] == ["0000_a.png", "0001_b.png"]
