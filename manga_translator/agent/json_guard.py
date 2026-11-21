"""Pages the user saved in the editor while an AI Agent run was working on them.

The editor and the run share each page's JSON: the run reads a page, works on it for a while and
writes it back, so a save in the editor in between would be lost. Every step of the run that writes
a page's JSON (the translator, proofreading, the font check, the inpaint review, the render) first
checks that the file is still what the run last read or wrote (``taken``). When it is not, the user
saved it: the page is theirs for the rest of the run. The step keeps the user's file, the
translator puts its own version in ``agent/conflicts/`` (``keep_agent_version``) and flags the
page, and the later steps leave it alone.

Only files the run has noted (``note``) are checked, so code outside a run works as before; the
workflow forgets everything when the run ends (``reset``).
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

from .store import write_json_atomic

_lock = threading.Lock()
_seen: Dict[str, Optional[Tuple[int, int]]] = {}
_users: Set[str] = set()

REASONS = {
    "CHT": {"during": "你在 AI Agent 處理這頁時用編輯器存了檔：保留你的版本，之後的步驟不再改這頁。AI Agent 的版本另存在 {file}",
            "later": "你在 AI Agent 執行中用編輯器存了這頁：保留你的版本，校對、字型檢查、修復審查和渲染都跳過這頁"},
    "CHS": {"during": "你在 AI Agent 处理这页时用编辑器存了档：保留你的版本，之后的步骤不再改这页。AI Agent 的版本另存在 {file}",
            "later": "你在 AI Agent 运行中用编辑器存了这页：保留你的版本，校对、字体检查、修复审查和渲染都跳过这页"},
    "ENG": {"during": "You saved this page in the editor while the AI Agent was working on it: your version stays and "
                      "the later steps leave the page alone. The AI Agent's version is in {file}",
            "later": "You saved this page in the editor during the AI Agent run: your version stays; proofreading, "
                     "the font check, the inpaint review and the render skip the page"},
}


def reason(lang: str, kind: str, **fields: Any) -> str:
    """The review-list reason in the target language (English for the others)."""
    texts = REASONS.get(lang) or REASONS["ENG"]
    return texts[kind].format(**fields)


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _stat(path: str) -> Optional[Tuple[int, int]]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def note(path: Optional[str]) -> None:
    """The run read or wrote ``path``: the file as it is now is the run's."""
    if not path:
        return
    key = _key(path)
    with _lock:
        if key not in _users:
            _seen[key] = _stat(path)


def taken(path: Optional[str]) -> bool:
    """The user saved ``path`` since the run last read or wrote it: the page is theirs now."""
    if not path:
        return False
    key = _key(path)
    with _lock:
        if key in _users:
            return True
        if key not in _seen or _stat(path) == _seen[key]:
            return False
        _users.add(key)
        return True


def users_files() -> Set[str]:
    """The JSON files the user took during the run (normalized paths)."""
    with _lock:
        return set(_users)


def reset() -> None:
    with _lock:
        _seen.clear()
        _users.clear()


def keep_agent_version(json_path: str, data: Any, agent_dir: str) -> str:
    """The run's version of a page the user took, in ``<agent_dir>/conflicts``; returns its path."""
    folder = os.path.join(agent_dir, "conflicts")
    name, ext = os.path.splitext(os.path.basename(json_path))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = os.path.join(folder, f"{name}.{stamp}{ext}")
    number = 1
    while os.path.exists(target):
        number += 1
        target = os.path.join(folder, f"{name}.{stamp}~{number}{ext}")
    write_json_atomic(target, data)
    return target
