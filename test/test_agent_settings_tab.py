"""Settings page: the "AI Agent" tab lists the agent options with labels and descriptions in every locale."""
import _bootstrap  # noqa: I001
import json
import re

from manga_translator.agent.settings import AgentSettings

from core.config_models import AppSettings
from services.i18n_service import I18nManager
from ui.main_page.dynamic_settings import _setting_control_kind

ROOT = _bootstrap.ROOT
LOCALE_DIR = ROOT / "desktop_qt_ui" / "locales"
LOCALES = ("zh_CN", "zh_TW", "en_US", "ja_JP", "ko_KR", "es_ES")
LAYOUT = json.loads((ROOT / "desktop_qt_ui" / "ui" / "main_page" / "settings_tab_layout.json").read_text(encoding="utf-8"))


def _agent_tab():
    return next(tab for tab in LAYOUT["tabs"] if tab["id"] == "tab_agent")


def _agent_keys():
    return [item.split(".", 1)[1] for item in _agent_tab()["items"] if isinstance(item, str)]


def test_agent_tab_lists_real_options_once_with_a_control():
    tab = _agent_tab()
    titles = [t["title"] for t in LAYOUT["tabs"]]
    assert titles.index("AI Agent") == titles.index("Mode Specific") - 1
    keys = _agent_keys()
    assert "proofread" in keys and "inpaint_review" in keys
    defaults = AppSettings().model_dump()["agent"]
    for key in keys:
        assert key in AgentSettings.model_fields, key
        full_key = f"agent.{key}"
        assert sum(t["items"].count(full_key) for t in LAYOUT["tabs"]) == 1, full_key
        assert _setting_control_kind(full_key, key, defaults[key], [], {}) is not None, key
    # the switches come first, then the "Advanced" divider and the numbers and text
    divider = next(i for i, item in enumerate(tab["items"]) if isinstance(item, dict))
    assert tab["items"][divider] == {"kind": "divider", "title": "Advanced"}
    assert all(isinstance(defaults[k.split(".", 1)[1]], bool) for k in tab["items"][:divider])
    assert AgentSettings().proofread is False


def test_agent_labels_are_mapped_and_translated_in_every_locale():
    source = (ROOT / "desktop_qt_ui" / "app_logic.py").read_text(encoding="utf-8")
    catalogs = {
        locale: json.loads((LOCALE_DIR / f"{locale}.json").read_text(encoding="utf-8")) for locale in LOCALES
    }
    keys = _agent_keys()
    for key in keys:
        assert len(re.findall(rf'"{key}": self\._t\("label_{key}"\)', source)) == 1, key
    names = ["AI Agent"] + [f"label_{k}" for k in keys] + [f"desc_agent_{k}" for k in keys]
    for locale, catalog in catalogs.items():
        manager = I18nManager(locale_dir=str(LOCALE_DIR), fallback_locale="zh_CN", config_language=locale)
        for name in names:
            assert catalog.get(name, "").strip(), (locale, name)
            assert manager.translate(name) == catalog[name], (locale, name)
    for locale in ("en_US", "ja_JP", "ko_KR", "es_ES"):
        for key in keys:
            assert catalogs[locale][f"desc_agent_{key}"] != catalogs["zh_CN"][f"desc_agent_{key}"], (locale, key)
            assert catalogs[locale][f"desc_agent_{key}"] != catalogs["zh_TW"][f"desc_agent_{key}"], (locale, key)
