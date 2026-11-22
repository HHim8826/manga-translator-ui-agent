"""A PDF page is taken as its embedded image only when that image is the whole page (the project review
of 2026-11-22, P2-03: a page with two small images and its dialogue set as text gave two blank small
images, and the page with the dialogue was not there)."""
import io

import fitz
from PIL import Image

from desktop_qt_ui.utils.archive_extractor import extract_images_from_pdf


def _png(size, color):
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, format="PNG")
    return data.getvalue()


def _pdf(path, build):
    doc = fitz.open()
    for make in build:
        make(doc.new_page(width=200, height=300))
    doc.save(str(path))
    doc.close()
    return str(path)


def _scan(page):  # a scanned page: one image over the whole page
    page.insert_image(page.rect, stream=_png((800, 1200), "red"))


def _scan_with_ocr(page):  # the same with an invisible OCR text layer
    _scan(page)
    page.insert_text((20, 50), "OCR text", render_mode=3)


def _mixed(page):  # two small pictures and the dialogue as text
    page.insert_image(fitz.Rect(10, 10, 60, 60), stream=_png((50, 50), "white"))
    page.insert_image(fitz.Rect(100, 10, 150, 60), stream=_png((50, 50), "white"))
    page.insert_text((20, 200), "DIALOGUE HERE", fontsize=20)


def _text_only(page):
    page.insert_text((20, 100), "TEXT", fontsize=30)


def _turned(page):
    page.insert_image(page.rect, stream=_png((1200, 800), "blue"), rotate=90)


def test_each_page_gives_one_image_and_text_is_kept(tmp_path):
    pdf = _pdf(tmp_path / "book.pdf", [_scan, _mixed, _scan_with_ocr, _text_only, _turned])
    images = extract_images_from_pdf(pdf, str(tmp_path / "out"))
    assert len(images) == 5  # one per page, in order
    sizes = [Image.open(p).size for p in images]
    assert sizes[0] == (800, 1200) and sizes[2] == (800, 1200)  # the scans: their own image, as it is
    assert Image.open(images[0]).getpixel((5, 5)) == (255, 0, 0)
    # the mixed page is rendered whole: the page's size, and the dialogue is in it
    mixed = Image.open(images[1]).convert("L")
    assert mixed.size == (400, 600)  # 2x: its pictures are 1:1
    text_box = mixed.crop((40, 360, 400, 420))
    assert text_box.getextrema()[0] < 100  # dark text where the dialogue is
    assert images[1].endswith(".png") and images[3].endswith(".png")
    assert Image.open(images[4]).size == (800, 1200)  # turned: rendered at the image's resolution
