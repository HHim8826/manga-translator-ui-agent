"""Review 2026-11-28 X1: the AI Agent workflow always uses the OpenAI-compatible API (OPENAI_API_KEY,
OPENAI_API_BASE), but the check before a run and the API Management page went by
``translator.translator``: with ``gemini_hq`` a run asked for a Gemini key the agent never uses and
the page hid the OpenAI fields; with ``none`` a run started without any key and failed after the
scan. No ConfigService is made: it reads and rewrites the real configuration files."""
import _bootstrap  # noqa: F401, I001

from types import SimpleNamespace

from core.config_models import AGENT_TRANSLATOR, AppSettings, api_translator
from services.config_service import ConfigService
from ui.main_page.dynamic_settings import _selected_api_group_keys
from ui.main_page.env_management import _get_current_translator_key


def _config(translator: str, agent: bool) -> AppSettings:
    config = AppSettings()
    config.translator.translator = translator
    config.cli.agent_workflow = agent
    return config


def _missing(config: AppSettings, env: dict) -> list:
    service = SimpleNamespace(load_env_vars=lambda: {}, _has_env_value=ConfigService._has_env_value)
    return [item["selected_value"] for item in ConfigService.get_missing_runtime_api_requirements(service, config, env)]


def test_the_agent_run_asks_for_the_openai_key_whatever_the_translator():
    assert api_translator(_config("gemini_hq", True)) == AGENT_TRANSLATOR
    assert api_translator(_config("gemini_hq", False)) == "gemini_hq"
    assert _missing(_config("gemini_hq", True), {"OPENAI_API_KEY": "sk-test"}) == []
    assert _missing(_config("gemini_hq", True), {}) == ["openai"]
    assert _missing(_config("none", True), {}) == ["openai"]
    assert _missing(_config("gemini_hq", False), {"OPENAI_API_KEY": "sk-test"}) == ["gemini_hq"]  # as before


def test_the_api_page_shows_the_openai_group_for_the_agent():
    assert _selected_api_group_keys(_config("gemini_hq", True))["translation"] == ["translator_openai"]
    assert _selected_api_group_keys(_config("none", True))["translation"] == ["translator_openai"]
    assert _selected_api_group_keys(_config("sakura", False))["translation"] == ["translator_sakura"]


def test_the_api_tests_and_the_candidate_check_use_the_openai_api_for_the_agent():
    combo = SimpleNamespace(currentData=lambda: "gemini_hq")
    for agent, expected in ((True, AGENT_TRANSLATOR), (False, "gemini_hq")):
        config = _config("gemini_hq", agent)
        view = SimpleNamespace(env_translation_feature_combo=combo,
                               controller=SimpleNamespace(config_service=SimpleNamespace(get_config=lambda c=config: c)))
        assert _get_current_translator_key(view) == expected
