"""What an earlier AI Agent run left in a chapter, so a new run does not silently redo it.

The preprocess of a run writes every page's JSON again, so running the workflow on a chapter the
agent already did replaces its translations, and the user's changes from the editor with them.
Before a run, the app asks when pages already have an agent result (``previous_states``): the user
can process only the pages that are not done, or redo everything.

A page counts as edited when its JSON differs from what the workflow saved last: the workflow keeps
a fingerprint of each page's JSON at its end (``agent/saved_json.json``). Runs made before the
fingerprints existed fall back on the time: the JSON was written more than ``EDIT_MARGIN`` seconds
after the chapter's last run log entry. Before the preprocess replaces an edited JSON, the workflow
copies it to ``agent/backups/``.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .store import AgentStore, _read_json, write_json_atomic

logger = logging.getLogger("manga_translator.agent")

EDIT_MARGIN = 30.0
FINGERPRINTS = "saved_json.json"


@dataclass
class PageState:
    path: str
    status: str
    """The agent's status of the page (``done``, ``incomplete``, ``error``, ``stopped``); empty when
    no run has seen it."""
    edited: bool = False
    """The JSON changed after the workflow saved it (in the editor, most likely)."""
    json_path: str = ""

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]


def _store(image_path: str) -> AgentStore:
    from ..utils.path_manager import get_work_dir

    return AgentStore(get_work_dir(image_path))


def _fingerprint(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprints_path(store: AgentStore) -> str:
    return os.path.join(store.agent_dir, FINGERPRINTS)


def previous_states(paths: Sequence[str]) -> List[PageState]:
    """Every page's state from the earlier runs of its chapter."""
    from ..utils.path_manager import find_json_path

    stores: Dict[str, AgentStore] = {}
    prints: Dict[str, dict] = {}
    log_times: Dict[str, Optional[float]] = {}
    states = []
    for path in paths:
        store = _store(path)
        key = store.work_dir
        if key not in stores:
            stores[key] = store
            prints[key] = _read_json(_fingerprints_path(store), {}) if os.path.isdir(store.agent_dir) else {}
            latest = store.latest_run_log() if os.path.isdir(store.agent_dir) else None
            log_times[key] = os.path.getmtime(latest) if latest else None
        stem = os.path.splitext(os.path.basename(path))[0]
        side = store.load_sidecar(stem) if os.path.isdir(store.pages_dir) else None
        status = str((side or {}).get("status") or "")
        json_path = find_json_path(path) or ""
        edited = False
        if status and json_path and os.path.exists(json_path):
            saved = prints[key].get(stem)
            if saved:
                edited = _fingerprint(json_path) != saved
            elif log_times[key] is not None:
                edited = os.path.getmtime(json_path) > log_times[key] + EDIT_MARGIN
        states.append(PageState(path, status, edited, json_path))
    return states


def record_saved(paths: Sequence[str]) -> None:
    """Fingerprints of the pages' JSON as the workflow leaves them."""
    from ..utils.path_manager import find_json_path

    by_store: Dict[str, List[str]] = {}
    for path in paths:
        by_store.setdefault(_store(path).work_dir, []).append(path)
    for work_dir, pages in by_store.items():
        store = AgentStore(work_dir)
        target = _fingerprints_path(store)
        prints = _read_json(target, {})
        for path in pages:
            json_path = find_json_path(path)
            if json_path and os.path.exists(json_path):
                prints[os.path.splitext(os.path.basename(path))[0]] = _fingerprint(json_path)
        try:
            write_json_atomic(target, prints)
        except OSError:
            logger.warning(f"agent: could not save {target}", exc_info=True)


def back_up_edited(states: Sequence[PageState]) -> List[str]:
    """Copy the JSON of edited pages to ``agent/backups/`` before the preprocess replaces them;
    returns the copies."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    copies = []
    for state in states:
        if not (state.edited and state.json_path and os.path.exists(state.json_path)):
            continue
        folder = os.path.join(_store(state.path).agent_dir, "backups")
        os.makedirs(folder, exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(state.json_path))
        target = os.path.join(folder, f"{name}.{stamp}{ext}")
        shutil.copy2(state.json_path, target)
        copies.append(target)
    return copies
