"""Agent settings that would make the workflow useless are refused (review 18): typed on the settings
page, loaded from config.json and imported from a file, with the same check and the same hints."""
import _bootstrap  # noqa: I001
import json
import logging
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from manga_translator.agent.settings import MIN_CONTEXT_TOKENS, AgentSettings

from core.config_models import AppSettings
from core.setting_checks import check_change, describe, problems
from services.i18n_service import I18nManager
from test_agents_page import app  # noqa: F401  (one QApplication for the session)

LOCALE_DIR = _bootstrap.ROOT / "desktop_qt_ui" / "locales"
ZH_TW = json.loads((LOCALE_DIR / "zh_TW.json").read_text(encoding="utf-8"))


def _tw():
    return I18nManager(locale_dir=str(LOCALE_DIR), fallback_locale="zh_CN", config_language="zh_TW").translate


def _label(path):
    return ZH_TW[f"label_{path.rsplit('.', 1)[-1]}"]


@pytest.mark.parametrize("values", [
    {"max_context_tokens": 0},
    {"max_context_tokens": MIN_CONTEXT_TOKENS - 1},
    {"compact_threshold_tokens": 0},
    {"compact_threshold_tokens": 400_000},  # not below the default limit
    {"max_context_tokens": 200_000},  # below the default compaction threshold
    {"low_confidence_threshold": 101},
    {"low_confidence_threshold": -0.1},
    {"request_timeout": 0},
])
def test_the_model_refuses_values_that_break_the_workflow(values):
    with pytest.raises(ValidationError):
        AgentSettings(**values)
    with pytest.raises(ValidationError):  # the whole config model too
        AppSettings.model_validate({"agent": values})


def test_the_model_takes_sound_values():
    s = AgentSettings(max_context_tokens=200_000, compact_threshold_tokens=150_000, low_confidence_threshold=1)
    assert (s.max_context_tokens, s.compact_threshold_tokens, s.low_confidence_threshold) == (200_000, 150_000, 1)
    assert AgentSettings(low_confidence_threshold=0).low_confidence_threshold == 0
    AppSettings()  # the defaults pass


@pytest.mark.parametrize("path, value, hint", [
    ("agent.max_context_tokens", 0, "上下文上限（token）至少要 32000。"),
    ("agent.compact_threshold_tokens", 400_000, "壓縮門檻（token）必須小於上下文上限 400000。"),
    ("agent.low_confidence_threshold", 101.0, "低信心門檻必須在 0 到 1 之間。"),
    ("agent.low_confidence_threshold", "abc", "低信心門檻必須是數字。"),
    ("agent.max_context_tokens", "1.5", "上下文上限（token）必須是整數。"),
    ("agent.keep_images_last_n_pages", 0, f"{ZH_TW['label_keep_images_last_n_pages']}至少要 1。"),
])
def test_a_change_gets_a_hint_in_the_users_language(path, value, hint):
    problem, _ = check_change(AppSettings(), path, value)
    assert problem is not None
    assert describe(_tw(), problem, _label(problem.path)) == hint


def test_lowering_the_limit_below_the_threshold_names_the_threshold():
    problem, _ = check_change(AppSettings(), "agent.max_context_tokens", 200_000)
    assert problem.path == "agent.compact_threshold_tokens"
    assert describe(_tw(), problem, _label(problem.path)) == "壓縮門檻（token）必須小於上下文上限 200000。"


def test_an_empty_field_means_the_default_and_text_becomes_a_number():
    assert check_change(AppSettings(), "agent.max_context_tokens", None) == (None, 400_000)
    assert check_change(AppSettings(), "agent.low_confidence_threshold", "0.5") == (None, 0.5)


def test_an_older_problem_elsewhere_does_not_block_a_change():
    config = AppSettings()
    config.agent.keep_images_last_n_pages = 0  # stored before this check existed
    assert check_change(config, "agent.max_steps_per_page", 10) == (None, 10)
    problem, _ = check_change(config, "agent.max_steps_per_page", 0)
    assert problem.path == "agent.max_steps_per_page"


def _view(config):
    sent = []
    t = _tw()
    view = SimpleNamespace(config_service=SimpleNamespace(get_config=lambda: config), _t=t,
                           setting_changed=SimpleNamespace(emit=lambda *a: sent.append(a)),
                           _translated_setting_label=_label)
    return view, sent


def test_the_settings_field_turns_red_and_keeps_the_saved_value(app):  # noqa: F811
    from ui.main_page.dynamic_settings import QLineEdit, _on_numeric_input_changed

    view, sent = _view(AppSettings())
    field = QLineEdit("400000")
    field.setText("0")
    _on_numeric_input_changed(view, field.text(), "agent.max_context_tokens", int, field)
    assert sent == [] and field.isError()
    assert field.toolTip() == "上下文上限（token）至少要 32000。\n沒有儲存，仍使用 400000。"

    field.setText("120000")
    _on_numeric_input_changed(view, field.text(), "agent.max_context_tokens", int, field)
    assert sent == [] and "壓縮門檻（token）必須小於上下文上限 120000。" in field.toolTip()

    field.setText("500000")
    _on_numeric_input_changed(view, field.text(), "agent.max_context_tokens", int, field)
    assert sent == [("agent.max_context_tokens", 500_000)]
    assert not field.isError() and field.toolTip() == ""

    field.setText("")  # empty: the default, shown in the field
    _on_numeric_input_changed(view, field.text(), "agent.max_context_tokens", int, field)
    assert sent[-1] == ("agent.max_context_tokens", 400_000) and field.text() == "400000"

    view, sent = _view(AppSettings())
    _on_numeric_input_changed(view, "101", "agent.low_confidence_threshold", float, field)
    assert sent == [] and field.toolTip().startswith("低信心門檻必須在 0 到 1 之間。")


def test_a_red_field_is_saved_once_the_field_it_depends_on_allows_it(app):  # noqa: F811
    """Review 2026-11-28 X3: the limit lowered first (refused: under the threshold), then the threshold
    lowered under it; the limit field still showed 128000 in red while 400000 stayed saved."""
    from ui.main_page.dynamic_settings import QLineEdit, _on_numeric_input_changed

    config = AppSettings()
    view, sent = _view(config)
    view.setting_changed = SimpleNamespace(
        emit=lambda key, value: sent.append((key, value)) or setattr(config.agent, key.split(".")[-1], value))
    limit, threshold = QLineEdit("400000"), QLineEdit("300000")
    limit.setText("128000")
    _on_numeric_input_changed(view, limit.text(), "agent.max_context_tokens", int, limit)
    assert sent == [] and limit.isError()
    threshold.setText("100000")
    _on_numeric_input_changed(view, threshold.text(), "agent.compact_threshold_tokens", int, threshold)
    assert sent == [("agent.compact_threshold_tokens", 100_000), ("agent.max_context_tokens", 128_000)]
    assert not limit.isError() and limit.toolTip() == "" and config.agent.max_context_tokens == 128_000
    # one that is still wrong stays red and unsaved
    limit.setText("90000")
    _on_numeric_input_changed(view, limit.text(), "agent.max_context_tokens", int, limit)
    threshold.setText("95000")
    _on_numeric_input_changed(view, threshold.text(), "agent.compact_threshold_tokens", int, threshold)
    assert limit.isError() and config.agent.max_context_tokens == 128_000
    assert sent[-1] == ("agent.compact_threshold_tokens", 95_000)


def test_the_settings_page_still_saves_optional_fields_empty(app):  # noqa: F811
    from ui.main_page.dynamic_settings import QLineEdit, _on_numeric_input_changed

    view, sent = _view(AppSettings())
    field = QLineEdit("")
    _on_numeric_input_changed(view, "", "render.font_size", int, field)
    assert sent == [("render.font_size", None)] and not field.isError()


def _load(tmp_path, agent):
    from services.config_service import ConfigService

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agent": agent}), encoding="utf-8")
    fake = SimpleNamespace(logger=logging.getLogger("test"), config_changed=SimpleNamespace(emit=lambda *_: None))
    assert ConfigService.load_config_file(fake, str(path))
    return fake.current_config.agent


def test_loading_keeps_a_lower_limit_with_its_lower_threshold(tmp_path):
    """The keys are checked one by one; the limit comes before the threshold that makes it valid."""
    agent = _load(tmp_path, {"max_context_tokens": 200_000, "compact_threshold_tokens": 150_000})
    assert (agent.max_context_tokens, agent.compact_threshold_tokens) == (200_000, 150_000)


def test_loading_replaces_unusable_values_with_the_defaults(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        agent = _load(tmp_path, {"max_context_tokens": 0, "low_confidence_threshold": 101, "max_steps_per_page": 12})
    assert (agent.max_context_tokens, agent.low_confidence_threshold, agent.max_steps_per_page) == (400_000, 0.7, 12)
    assert "agent.max_context_tokens" in caplog.text and "agent.low_confidence_threshold" in caplog.text


def _import(tmp_path, monkeypatch, agent):
    import PyQt6.QtWidgets as widgets
    from app_logic import MainAppLogic

    path = tmp_path / "import.json"
    path.write_text(json.dumps({"agent": agent}), encoding="utf-8")
    shown, stored = [], []
    monkeypatch.setattr(widgets.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), "")))
    for kind in ("warning", "information", "critical"):
        monkeypatch.setattr(widgets.QMessageBox, kind,
                            staticmethod(lambda _p, title, text, kind=kind: shown.append((kind, title, text))))
    labels = {key[len("label_"):]: text for key, text in ZH_TW.items() if key.startswith("label_")}
    logic = SimpleNamespace(
        _t=_tw(), logger=logging.getLogger("test"), get_display_mapping=lambda key: labels if key == "labels" else None,
        config_service=SimpleNamespace(get_config=AppSettings, set_config=stored.append, save_config_file=lambda: None),
        config_loaded=SimpleNamespace(emit=lambda *_: None))
    logic._setting_problem_hints = lambda d: MainAppLogic._setting_problem_hints(logic, d)
    MainAppLogic.import_config(logic)
    return shown, stored, str(path)


def test_an_import_with_unusable_values_is_refused_with_field_hints(tmp_path, monkeypatch):
    shown, stored, path = _import(tmp_path, monkeypatch, {"max_context_tokens": 0, "low_confidence_threshold": 101})
    assert stored == []
    [(kind, title, text)] = shown
    assert kind == "warning" and title == "匯入失敗"
    assert text == (f"{path} 裡有不能用的值，所以沒有匯入任何設定：\n\n"
                    "• 上下文上限（token）至少要 32000。\n• 低信心門檻必須在 0 到 1 之間。")


def test_a_sound_import_goes_through(tmp_path, monkeypatch):
    shown, stored, _ = _import(tmp_path, monkeypatch, {"max_context_tokens": 200_000, "compact_threshold_tokens": 150_000})
    assert [s.agent.max_context_tokens for s in stored] == [200_000]
    assert [kind for kind, *_ in shown] == ["information"]


def test_problems_lists_every_bad_key():
    found = problems({"agent": {"max_context_tokens": 0, "low_confidence_threshold": 2}})
    assert {p.path for p in found} == {"agent.max_context_tokens", "agent.low_confidence_threshold"}
