"""No thin fonts (the user, chapter 36): thin fonts are measured from their drawn strokes and left off
the font sheet, missing characters are not borrowed from them, and a style from an earlier chapter
that used one moves to the sheet font that looks most like it."""
import os

import pytest
from fontTools.ttLib import TTCollection, TTFont

from manga_translator.agent import fonts
from manga_translator.agent.events import EventLog
from manga_translator.agent.fonts import MIN_STROKE, FontCatalog, FontFace, FontService
from manga_translator.agent.settings import AgentSettings
from manga_translator.agent.store import AgentStore
from manga_translator.utils import BASE_PATH
from test_agent_core import AgentRunner, ScriptedClient, make_page
from test_agent_fonts import SheetFonts

WINDOWS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
FILES = [os.path.join(WINDOWS, name) for name in ("mingliu.ttc", "msjh.ttc", "msjhbd.ttc", "simhei.ttf")]
FILES.append(os.path.join(BASE_PATH, "fonts", "msyh.ttc"))
needs_fonts = pytest.mark.skipif(not all(os.path.exists(p) for p in FILES), reason="Windows CJK fonts not found")


@pytest.fixture(scope="module")
def catalog():
    faces = []
    for path in FILES:
        collection = TTCollection(path, lazy=True).fonts if path.endswith(".ttc") else [TTFont(path, lazy=True)]
        for index, font in enumerate(collection):
            faces.extend(fonts._faces_of(font, path, index))
    return FontCatalog(faces)


@needs_fonts
def test_thin_fonts_are_told_apart_by_their_strokes(catalog):
    measured = {name: catalog.stroke(catalog.find(name), "cjk")
                for name in ("MingLiU", "Microsoft JhengHei", "Microsoft JhengHei::Bold", "SimHei", "Microsoft YaHei")}
    assert measured["MingLiU"] < measured["Microsoft JhengHei"] < MIN_STROKE  # MingLiU says weight 400
    assert MIN_STROKE < measured["SimHei"] < measured["Microsoft JhengHei::Bold"]
    assert catalog.thin(catalog.find("MingLiU"), "cjk") and not catalog.thin(catalog.find("Microsoft YaHei"), "cjk")
    nowhere = FontFace(path="missing.ttf", index=0, family="Gone", style="Regular", weight=400, names=("Gone",),
                       locales=frozenset())
    assert catalog.stroke(nowhere, "cjk") is None and not catalog.thin(nowhere, "cjk") and not catalog.thin(None, "cjk")


@needs_fonts
def test_the_sheet_and_the_borrowed_characters_have_no_thin_font(catalog):
    service = FontService(catalog, None, "Microsoft YaHei", "CHT")
    service.prepare_sheet()
    names = [face.value for face in service.choices]
    assert names and "MingLiU" not in names and "PMingLiU" not in names and "Microsoft JhengHei" not in names
    assert not [face for face in service.choices if catalog.thin(face, "cjk")]
    ranked = catalog.similar(catalog.find("Microsoft YaHei"), "值得", "CHT", limit=20)
    assert ranked and not [face for _, face in ranked if catalog.thin(face, "cjk")]
    # a thin font of the user's own may still borrow from a thin one
    thin_ref = catalog.similar(catalog.find("MingLiU"), "值得", "CHT", limit=20)
    assert any(catalog.thin(face, "cjk") for _, face in thin_ref)


@needs_fonts
def test_a_thin_style_font_gets_a_stand_in_from_the_sheet(catalog):
    service = FontService(catalog, None, "Microsoft YaHei", "CHT")
    service.prepare_sheet()
    stand_in = service.stand_in("MingLiU")
    assert stand_in in [face.value for face in service.choices]
    assert service.stand_in("SimHei") is None and service.stand_in("Not Installed") is None
    assert FontService(catalog, None, "MingLiU", "CHT").stand_in("PMingLiU") is None  # the user's font is thin


class ThinSheetFonts(SheetFonts):
    def stand_in(self, value):
        return {"MingLiU": "Noto Serif TC", "Yu Mincho::Light": ""}.get(value)


def test_styles_of_earlier_chapters_move_off_thin_fonts(tmp_path):
    image_path, _ = make_page(tmp_path)
    store = AgentStore(os.path.join(os.path.dirname(image_path), "manga_translator_work"))
    events = EventLog(store.run_log_path("r1"), "r1")
    work = {"明體旁白": {"font": "MingLiU", "reason": "明體", "chapter": "第35話"},
            "細明註解": {"font": "Yu Mincho::Light", "reason": "細字"},
            "手寫": {"font": "AR TianniuB5", "reason": "手寫"}}
    runner = AgentRunner(AgentSettings(recall_pass=False), ScriptedClient([]), store, events, [image_path], "CHT",
                         fonts=ThinSheetFonts(), font_styles=work)
    styles = runner.memory.font_styles
    assert styles["明體旁白"] == {"font": "Noto Serif TC", "reason": "明體", "page": "", "from": "第35話"}
    assert styles["細明註解"]["font"] == "" and styles["手寫"]["font"] == "AR TianniuB5"
    text = runner.preamble[1]["content"][0]["text"]
    assert "- 明體旁白: font 1 (Noto Serif TC) (明體)" in text and "MingLiU" not in text
    events.close()
