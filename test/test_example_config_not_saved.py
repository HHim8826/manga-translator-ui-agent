"""Review 2026-11-24 B5: running from source, every settings change was also written to
``config/config-example.json``, the repository's template, so the user's settings (the model, the
workflow switches) went into it and into ``git commit -a``. Only with
``MANGA_TRANSLATOR_SAVE_EXAMPLE_CONFIG=1`` now. No ConfigService is made: it reads and rewrites the
real configuration files."""
import _bootstrap  # noqa: F401, I001

import os
import threading
from types import SimpleNamespace

from services.config_service import SAVE_EXAMPLE_CONFIG_ENV, ConfigService


def _saved_paths(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(SAVE_EXAMPLE_CONFIG_ENV, raising=False)
    else:
        monkeypatch.setenv(SAVE_EXAMPLE_CONFIG_ENV, value)
    service = SimpleNamespace(
        user_config_path=os.path.abspath("cfg/config.json"),
        default_config_path=os.path.abspath("cfg/config-example.json"),
        _build_config_payload=lambda path: {"path": path},
        _write_lock=threading.Lock(), _pending_config_writes={}, config_path=None,
        _schedule_write=lambda: True, flush_pending_writes=lambda: True,
        logger=SimpleNamespace(error=lambda *a, **k: None))
    assert ConfigService.save_config_file(service) is True
    return sorted(os.path.basename(path) for path in service._pending_config_writes)


def test_the_settings_go_to_the_users_file_only(monkeypatch):
    assert _saved_paths(monkeypatch, None) == ["config.json"]
    assert _saved_paths(monkeypatch, "0") == ["config.json"]
    assert _saved_paths(monkeypatch, "1") == ["config-example.json", "config.json"]  # updating the template
