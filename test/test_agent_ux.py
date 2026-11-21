"""The AI Agent feature reads like the rest of the app: one name, the app's own words, results in the
interface language, totals that match the page list, and the translation page progress bar."""
import _bootstrap  # noqa: I001
import asyncio
import json
import os
import re
from types import SimpleNamespace

import pytest

from ui.main_page.pages import agent_results
from ui.main_page.pages.agent_results import flag_text, result_text

from test_agent_core import FakeTranslator, ScriptedClient, _finish_page

ROOT = _bootstrap.ROOT
LOCALE_DIR = ROOT / "desktop_qt_ui" / "locales"
LOCALES = ("zh_CN", "zh_TW", "en_US", "ja_JP", "ko_KR", "es_ES")
CATALOGS = {locale: json.loads((LOCALE_DIR / f"{locale}.json").read_text(encoding="utf-8")) for locale in LOCALES}


def _t(key, **kwargs):
    return key + (" " + " ".join(f"{k}={v}" for k, v in sorted(kwargs.items())) if kwargs else "")


def _tool_keys():
    from ui.main_page.pages.agents_page import TOOL_KEYS
    return TOOL_KEYS


from test_agents_page import app  # noqa: E402,F401  (one QApplication for the session)


# --- names and words -------------------------------------------------------------------
def test_one_name_and_the_apps_own_words():
    for locale, catalog in CATALOGS.items():
        assert catalog["Agents"] == catalog["AI Agent"] != "Agents", locale  # the navigation and the settings tab
    tw = CATALOGS["zh_TW"]
    agent_texts = [v for k, v in tw.items() if "agent" in k.lower() or k.startswith("Tip: Detection and local OCR")]
    for old in ("修圖", "去字", "嵌字", "偵測", "Agents 頁面"):
        assert not [v for v in agent_texts if old in v], old
    assert tw["Agents Render"] == tw["Render"] == "渲染" and tw["Agent Page Inpainting"] == "修復審查"
    assert not re.search(r"(?<![A-Za-z])agent(?![A-Za-z])", " ".join(agent_texts))  # always "Agent" in Chinese
    # the work prompt file is no longer only read
    assert "只讀不寫" not in tw["desc_agent_use_work_prompt"] and "新詞寫回作品提示詞" in tw["desc_agent_use_work_prompt"]
    assert "only read" not in CATALOGS["en_US"]["desc_agent_use_work_prompt"]


def _result_keys():
    source = (ROOT / "desktop_qt_ui" / "ui" / "main_page" / "pages" / "agent_results.py").read_text(encoding="utf-8")
    keys = set(re.findall(r'(?<![A-Za-z_])t\("([^"{}]+)"', source))  # t("..."), not split(",")
    keys |= {f"Agent Flag {k}" for k in agent_results.FLAG_KINDS}
    keys |= {f"Agent Skip {k}" for k in agent_results.SKIP_KINDS}
    keys |= {f"Agent Issue {k}" for k in agent_results.ISSUES}
    keys |= {f"Agent Mask {k}" for k in ("dilate", "erode", "paint add", "paint remove")}
    keys |= {"Agents Page Confidence", "Agent Progress Preprocess", "Agent Progress Translate",
             "Agent Progress Proofread", "Agent Progress Render"}
    return keys


def test_every_new_text_is_in_every_locale():
    keys = _result_keys()
    assert len(keys) > 50
    for locale, catalog in CATALOGS.items():
        missing = sorted(k for k in keys if not catalog.get(k, "").strip())
        assert not missing, (locale, missing)
    for key in keys:  # the placeholders are the same in every locale
        names = {locale: set(re.findall(r"{(\w+)}", CATALOGS[locale][key])) for locale in LOCALES}
        assert len({frozenset(v) for v in names.values()}) == 1, (key, names)


# --- results -------------------------------------------------------------------------------
@pytest.mark.parametrize("tool,summary,ok,expected", [
    ("get_candidates", "12 regions", True, "Agent Result Regions count=12"),
    ("zoom", "#6 ×2.54", True, "#6 ×2.54"),
    ("zoom", "area [1, 2, 30, 40] ×3", True, "Agent Result Zoom Area box=1, 2, 30, 40 scale=3"),
    ("get_neighbors", "none", True, "Agent Result No Neighbors"),
    ("get_neighbors", "previous page (-1), next page (+1)", True,
     "Agent Result Page Before n=1Agent Result SeparatorAgent Result Page After n=1"),
    ("skip_region", "#5 (sfx)", True, "Agent Result With Kind id=#5 kind=Agent Skip sfx"),
    ("set_text", "#1: …お → ……喔。", True, "#1: …お → ……喔。"),
    ("glossary_search", "ギルド: 2", True, "Agent Result Found count=2 query=ギルド"),
    ("character_get", "all: 4", True, "Agent Result Found count=4 query=Agent Result All"),
    ("glossary_upsert", "くえ → 咕欸 (added)", True, "Agent Result Added item=くえ → 咕欸"),
    ("character_upsert", "りょうた (updated)", True, "Agent Result Updated item=りょうた"),
    ("story_summary_update", "117 chars", True, "Agent Result Chars count=117"),
    ("flag_for_review", "#6 inpaint", True, "Agent Result With Kind id=#6 kind=Agent Flag inpaint"),
    ("flag_for_review", "001 #4", True, "001 #4"),  # the proofreader's: page and region
    ("mark_page_done", "confidence 0.99", True, "Agent Result Confidence percent=99"),
    ("mark_page_done", "missing #4, #5", False, "Agent Result Missing ids=#4Agent Result Separator#5"),
    ("mark_page_done", "1 個框譯文過長", False, "1 個框譯文過長"),
    ("budget_status", "91000 tokens", True, "Agent Result Tokens count=91000"),
    ("undo", "set_text #3", True, "Agent Result Undid what=Agent Tool set_text #3"),
    ("undo", "#10 erode 3px", True, "Agent Result Undid what=#10 Agent Mask erode px=3"),
    ("view_region", "#1 a1", True, "Agent Result Attempt id=#1 n=1"),
    ("view_region", "#4 M3 S5", True, "Agent Result Components id=#4 masks=3 strokes=5"),
    ("adjust_mask", "#6 dilate 7px (residual)", True, "#6 Agent Mask dilate px=7Agent Mask Residual Only"),
    ("mask_paint", "#10 paint remove D7:F7", True, "#10 Agent Mask paint remove cells=D7:F7"),
    ("mask_select", "#4 select +S5 -M2", True, "#4 Agent Mask select parts=+S5 -M2"),
    ("protect_lines", "#4 protect lines 2px", True, "#4 Agent Mask protect lines px=2"),
    ("inpaint_region", "#6 a2 lama_large: ok", True,
     "Agent Result Inpainted id=#6 issues=Agent Result No Issues method=lama_large n=2"),
    ("inpaint_region", "#4 a3 fill: residual_text, broken_lines", True,
     "Agent Result Inpainted id=#4 issues=Agent Issue residual_textAgent Result SeparatorAgent Issue broken_lines "
     "method=Agent Method fill n=3"),
    ("pick_candidate", "#6 a2", True, "Agent Result Picked id=#6 n=2"),
    ("pick_candidate", "#10 used its 15 operations: pick_candidate the best attempt", False,
     "Agent Result Out Of Operations id=#10 n=15"),
    ("finish_review", "done", True, "Agent Result Done"),
    ("finish_review", "residual #3, #5", False, "Agent Result Residual ids=#3Agent Result Separator#5"),
    ("revise_text", "unchanged", True, "Agent Result Unchanged"),
    ("finish_proofread", "25 revised", True, "Agent Result Revised count=25"),
    ("mask_paint", "cells must look like C4 or B3:D5 (grid of the last view_region)", False, "Agent Result Failed"),
    ("set_text", "bad arguments", False, "Agent Result Failed"),
    ("something_new", "anything", True, "anything"),
])
def test_results_in_the_interface_language(tool, summary, ok, expected):
    text, tip = result_text(_t, tool, summary, ok, _tool_keys())
    assert text == expected and tip == summary


def test_results_read_naturally_in_traditional_chinese():
    tw = CATALOGS["zh_TW"]

    def t(key, **kwargs):
        return tw[key].format(**kwargs)

    keys = _tool_keys()
    assert result_text(t, "mark_page_done", "missing #4, #5", False, keys)[0] == "#4、#5 還沒有結果"
    assert result_text(t, "inpaint_region", "#6 a2 lama_large: ok", True, keys)[0] == "#6 第 2 版（lama_large）：沒有問題"
    assert result_text(t, "skip_region", "#5 (sfx)", True, keys)[0] == "#5：音效"
    assert result_text(t, "undo", "merge #1+#2", True, keys)[0] == "復原：合併區域 #1+#2"
    assert flag_text(t, 4, "uncertain_ocr") == "#4：原文不確定" and flag_text(t, None, "step_limit") == "超過呼叫上限"


# --- the AI Agent page ------------------------------------------------------------------------
def test_finished_status_counts_every_agent(app):
    from ui.main_page.pages.agents_page import AgentsPage

    page = AgentsPage(_t)
    page.replay([
        {"type": "run_start", "pages": ["001.jpg", "002.jpg"], "model": "m"},
        {"type": "flag", "page": "001", "agent": "translator", "region": 2, "flag": "uncertain_ocr", "reason": "r"},
        {"type": "page_done", "page": "001", "status": "done", "flags": 1, "confidence": 0.75},
        {"type": "page_done", "page": "002", "status": "done", "flags": 0, "confidence": 0.99},
        {"type": "run_end", "pages": 2, "done": 2, "flagged": 1, "totals": {"requests": 10, "recall_requests": 2}},
        {"type": "inpaint_start", "page": "002", "agent": "inpaint", "units": 3, "issues": 1},
        {"type": "flag", "page": "002", "agent": "inpaint", "region": 3, "flag": "inpaint", "reason": "r"},
        {"type": "flag", "page": "002", "agent": "inpaint", "region": 4, "flag": "inpaint", "reason": "r"},
        {"type": "inpaint_done", "page": "002", "agent": "inpaint", "units": 3, "retried": 1, "flagged": 2,
         "confidence": 0.75, "flags": 2, "usage": {"requests": 5}},
        {"type": "proofread_start", "agent": "proofread", "pages": 2, "regions": 9},
        {"type": "flag", "page": "001", "agent": "proofread", "region": 5, "flag": "uncertain_translation",
         "reason": "r"},
        {"type": "proofread_done", "agent": "proofread", "revised": 1, "flagged": 1, "usage": {"requests": 3}},
    ])
    pages = [page.page_list.item(i).text() for i in range(1, page.page_list.count())]
    assert pages == ["001 · Agent Page Done · Agents Page Confidence percent=75 · Agents Flag Count count=2",
                     "002 · Agent Page Done · Agents Page Confidence percent=75 · Agents Flag Count count=2"]
    # 4 flags as the page list shows them; 10 + 2 recall + 5 inpaint review + 3 proofreading requests
    assert page.status_label.text() == "Agents Finished done=2 flagged=4 pages=2 requests=20"
    flag_rows = [page.timeline.topLevelItem(i) for i in range(page.timeline.topLevelItemCount())
                 if page.timeline.topLevelItem(i).text(2) == "Agents Flag"]
    assert flag_rows[0].text(3) == "Agent Result With Kind id=#2 kind=Agent Flag uncertain_ocr"


# --- the translation page progress bar -----------------------------------------------------------
class ReportingTranslator(FakeTranslator):
    """Reports batch progress like MangaTranslator, one page at a time."""

    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self._progress_hooks = []

    async def translate_batch(self, images_with_configs, save_info=None):
        result = await super().translate_batch(images_with_configs, save_info)
        total = len(images_with_configs)
        for done in range(1, total + 1):
            for hook in list(self._progress_hooks):
                await hook(f"batch:{done}:{done}:{total}:0:0", False)
        return result


def test_workflow_reports_each_stage(tmp_path):
    from manga_translator.agent.workflow import run_agent_workflow

    folder = tmp_path / "chapter"
    folder.mkdir()
    paths = [str(folder / f"00{i}.png") for i in (1, 2)]

    class Cfg:
        class translator:
            target_lang = "CHT"
            high_quality_prompt_path = ""

    calls = []
    translator = ReportingTranslator(tmp_path)
    asyncio.run(run_agent_workflow(translator, [(p, Cfg) for p in paths], {}, client=ScriptedClient(
        _finish_page() + _finish_page()), log=lambda m: None, progress=lambda *a: calls.append(a),
        agent_settings={"inpaint_review": False, "proofread": True, "choose_fonts": False}))
    assert calls == [("preprocess", 0, 2), ("preprocess", 1, 2), ("preprocess", 2, 2),
                     ("translate", 0, 2), ("translate", 1, 2), ("translate", 2, 2),
                     ("proofread", 2, 2),
                     ("render", 0, 2), ("render", 1, 2), ("render", 2, 2)]
    assert translator._progress_hooks == []  # the workflow's hook is gone afterwards


def test_progress_bar_text_names_the_stage():
    from app_logic import TranslationWorker

    tw = CATALOGS["zh_TW"]
    worker = SimpleNamespace(_t=lambda key, **kw: tw[key].format(**kw),
                             _AGENT_PROGRESS_KEYS=TranslationWorker._AGENT_PROGRESS_KEYS)
    text = TranslationWorker._agent_progress_text
    assert text(worker, "preprocess", 3, 11) == "前處理 3/11 頁"
    assert text(worker, "translate", 4, 11) == "AI Agent 翻譯 4/11 頁"
    assert text(worker, "proofread", 11, 11) == "整話校對"
    assert text(worker, "render", 2, 11) == "渲染 2/11 頁"
    assert text(worker, "unknown", 0, 1) == ""
    source = (ROOT / "desktop_qt_ui" / "app_logic.py").read_text(encoding="utf-8")
    assert "progress=agent_progress" in source and 'if progress_context.get("agent"):' in source
