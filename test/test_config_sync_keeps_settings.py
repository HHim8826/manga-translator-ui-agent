"""Review 2026-11-28 B1: at start the user's ``config.json`` kept only the keys of
``config/config-example.json``. A clean checkout's template has no ``agent`` section and no
``cli.agent_workflow``, so every start dropped the AI Agent settings; the release package's lacked
three ``agent`` keys. Now every setting of the settings model stays too. No ConfigService is made:
it reads and rewrites the real configuration files."""
import _bootstrap  # noqa: F401, I001

import json
from types import SimpleNamespace

from core.config_models import AppSettings
from services.config_service import ConfigService, sync_template


def _synced(tmp_path, example: dict, user: dict) -> dict:
    example_path, user_path = tmp_path / "config-example.json", tmp_path / "config.json"
    example_path.write_text(json.dumps(example), encoding="utf-8")
    user_path.write_text(json.dumps(user), encoding="utf-8")
    service = SimpleNamespace(default_config_path=str(example_path), user_config_path=str(user_path),
                              logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                                                     error=lambda *a, **k: None))
    service._sync_dict = lambda template, data: ConfigService._sync_dict(service, template, data)
    ConfigService._sync_user_config(service)
    return json.loads(user_path.read_text(encoding="utf-8"))


def test_settings_missing_from_the_template_are_kept(tmp_path):
    example = {"cli": {"verbose": False}, "translator": {"translator": "openai"}}  # no agent at all
    user = {"cli": {"verbose": True, "agent_workflow": True, "gone": 1},
            "agent": {"model": "my-model", "whole_chapter_conversation": True, "gone": 2},
            "translator": {"translator": "sakura"}, "old_section": {"x": 1}}
    synced = _synced(tmp_path, example, user)
    assert synced["cli"]["verbose"] is True and synced["cli"]["agent_workflow"] is True
    assert synced["agent"]["model"] == "my-model" and synced["agent"]["whole_chapter_conversation"] is True
    assert synced["translator"]["translator"] == "sakura"
    # what neither the template nor the settings model has still goes
    assert "gone" not in synced["cli"] and "gone" not in synced["agent"] and "old_section" not in synced
    # a setting the user never had comes with its default
    assert synced["agent"]["page_image_max_side"] == AppSettings().agent.page_image_max_side


def test_the_template_value_comes_before_the_models_default():
    template = sync_template({"agent": {"proofread": True}, "app": {"only_in_template": 3}})
    assert template["agent"]["proofread"] is True
    assert template["app"]["only_in_template"] == 3
    assert set(AppSettings().model_dump()) <= set(template)
    assert set(AppSettings().agent.model_dump()) <= set(template["agent"])
