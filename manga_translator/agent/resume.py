"""What an earlier run left in a chapter, and how a run keeps it safe.

The preprocess of a run writes every page's JSON again, so running the workflow on pages that already
have a translation replaces it, and the user's changes from the editor with it.

Before a run, the app asks when pages already have a result (``previous_states``): an AI Agent
result, or a translation the AI Agent did not make (another translation mode, the editor). The user
can process only the pages that are not done, or redo everything.

A page counts as edited when its JSON differs from what the workflow saved last: the workflow keeps
a fingerprint of each page's JSON at its end (``agent/saved_json.json``). Runs made before the
fingerprints existed fall back on the time: the JSON was written more than ``EDIT_MARGIN`` seconds
after the chapter's last run log entry.

During a run, ``RunGuard`` keeps the pages safe:

* before the preprocess, each page's JSON and sidecar are copied to ``agent/backups/``, and a journal
  (``agent/run_journal.json``) lists the pages the run has not finished yet;
* a page the preprocess did not write again (it failed there) is not given to the agent: its JSON is
  the earlier one, maybe for another image or other boxes;
* when the run stops or fails, the pages it did not finish get their JSON and sidecar back, so a page
  the agent never reached keeps its earlier translation and status;
* a journal left by a run that was killed is restored before the next run (``recover``).

Copies of the user's own work (edited after a run, or translated outside the AI Agent) stay in
``backups``; the copies of the agent's own results are removed when the run ends.
"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .store import AgentStore, _read_json, write_json_atomic

logger = logging.getLogger("manga_translator.agent")

EDIT_MARGIN = 30.0
FINGERPRINTS = "saved_json.json"
JOURNAL = "run_journal.json"


@dataclass
class PageState:
    path: str
    status: str
    """The agent's status of the page (``done``, ``incomplete``, ``error``, ``stopped``); empty when
    no run has seen it."""
    edited: bool = False
    """The JSON changed after the workflow saved it (in the editor, most likely)."""
    json_path: str = ""
    translated: bool = False
    """No AI Agent result, but the JSON has translations: another translation mode or the editor."""

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def users(self) -> bool:
        """The JSON holds the user's own work."""
        return self.edited or (not self.status and self.translated)


def _store(image_path: str) -> AgentStore:
    from ..utils.path_manager import get_work_dir

    return AgentStore(get_work_dir(image_path))


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _fingerprint(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path: str) -> Optional[list]:
    """(mtime, size, fingerprint) of a file; None when it does not exist."""
    try:
        stat = os.stat(path)
        return [stat.st_mtime_ns, stat.st_size, _fingerprint(path)]
    except OSError:
        return None


def _fingerprints_path(store: AgentStore) -> str:
    return os.path.join(store.agent_dir, FINGERPRINTS)


def _journal_path(store: AgentStore) -> str:
    return os.path.join(store.agent_dir, JOURNAL)


def has_translation(json_path: str) -> bool:
    """The JSON has at least one region with a translation."""
    try:
        raw = _read_json(json_path, {})
    except (OSError, ValueError):
        return False
    for entry in (raw.values() if isinstance(raw, dict) else []):
        regions = entry.get("regions", []) if isinstance(entry, dict) else entry
        for region in regions if isinstance(regions, list) else []:
            if isinstance(region, dict) and str(region.get("translation") or "").strip():
                return True
    return False


def previous_states(paths: Sequence[str]) -> List[PageState]:
    """Every page's state from the earlier runs of its chapter (a run that was killed is restored
    first)."""
    from ..utils.path_manager import find_json_path

    recover(paths)
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
        edited = translated = False
        exists = bool(json_path) and os.path.exists(json_path)
        if status and exists:
            saved = prints[key].get(stem)
            if saved:
                edited = _fingerprint(json_path) != saved
            elif log_times[key] is not None:
                edited = os.path.getmtime(json_path) > log_times[key] + EDIT_MARGIN
        elif exists:
            translated = has_translation(json_path)
        states.append(PageState(path, status, edited, json_path, translated))
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


def _copy_to(folder: str, source: str, name: str, stamp: str) -> str:
    """``source`` copied to ``folder/<name>.<stamp><ext>`` (``~2`` ... when that exists)."""
    os.makedirs(folder, exist_ok=True)
    base, ext = os.path.splitext(name)
    target = os.path.join(folder, f"{base}.{stamp}{ext}")
    number = 1
    while os.path.exists(target):
        number += 1
        target = os.path.join(folder, f"{base}.{stamp}~{number}{ext}")
    shutil.copy2(source, target)
    return target


def _remove(path: str) -> None:
    if path:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning(f"agent: could not remove {path}", exc_info=True)


def _restore(entry: dict) -> bool:
    """The page's JSON and sidecar as they were before the run; False when it had no JSON then (the
    run's partial work stays)."""
    if not entry.get("json_copy") or not os.path.exists(entry["json_copy"]):
        return False
    shutil.copy2(entry["json_copy"], entry["json"])
    sidecar = entry["sidecar"]
    if entry.get("sidecar_copy") and os.path.exists(entry["sidecar_copy"]):
        shutil.copy2(entry["sidecar_copy"], sidecar)
    elif os.path.exists(sidecar):  # made by this run
        os.remove(sidecar)
    return True


def _set_prints(store: AgentStore, prints_by_stem: Dict[str, Optional[str]]) -> None:
    target = _fingerprints_path(store)
    prints = _read_json(target, {})
    for stem, value in prints_by_stem.items():
        if value:
            prints[stem] = value
        else:
            prints.pop(stem, None)
    try:
        write_json_atomic(target, prints)
    except OSError:
        logger.warning(f"agent: could not save {target}", exc_info=True)


def _close(store: AgentStore, entries: Dict[str, dict], finished: Sequence[str]) -> List[str]:
    """Put back the pages of a journal that were not finished, drop the agent's own copies and the
    journal; returns the restored pages."""
    restored, prints = [], {}
    for stem, entry in entries.items():
        if stem in finished:
            continue
        try:
            if _restore(entry):
                restored.append(entry["image"])
                prints[stem] = entry.get("print")
        except OSError:
            logger.exception(f"agent: could not restore {entry.get('json')}")
            continue
    if prints:
        _set_prints(store, prints)
    for entry in entries.values():
        _remove(entry.get("sidecar_copy", ""))
        if entry.get("kind") == "agent":
            _remove(entry.get("json_copy", ""))
    _remove(_journal_path(store))
    return restored


def recover(paths: Sequence[str]) -> List[str]:
    """Restore the unfinished pages of a run that was killed (its journal is still there); returns
    them."""
    restored: List[str] = []
    seen = set()
    for path in paths:
        store = _store(path)
        if store.work_dir in seen:
            continue
        seen.add(store.work_dir)
        journal = _read_json(_journal_path(store), None) if os.path.isdir(store.agent_dir) else None
        if not isinstance(journal, dict):
            continue
        entries = {stem: e for stem, e in (journal.get("pages") or {}).items() if isinstance(e, dict)}
        pages = _close(store, entries, ())
        if pages:
            logger.warning(f"agent: an earlier run ended without finishing {len(pages)} page(s); "
                           f"their translations from before it are back: {', '.join(map(os.path.basename, pages))}")
        restored += pages
    return restored


class RunGuard:
    """Copies of the pages before the preprocess, and the journal of the pages the run has not
    finished (see the module doc)."""

    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        self.entries: Dict[str, dict] = {}
        """Normalized image path -> what the page had before the run."""
        self.stores: Dict[str, AgentStore] = {}
        self.finished: Dict[str, set] = {}
        """Work dir -> stems the run finished."""
        self.users_copies: List[str] = []
        """Copies of the user's own work, kept in ``backups``."""

    @classmethod
    def begin(cls, paths: Sequence[str]) -> "RunGuard":
        from ..utils.path_manager import find_json_path, get_json_path

        guard = cls(paths)
        states = previous_states(paths)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        journals: Dict[str, dict] = {}
        for state in states:
            store = _store(state.path)
            guard.stores.setdefault(store.work_dir, store)
            journal = journals.setdefault(store.work_dir, {"started": stamp, "pages": {},
                                                           "_prints": _read_json(_fingerprints_path(store), {})})
            json_path = state.json_path or find_json_path(state.path) or get_json_path(state.path, create_dir=False)
            sidecar = store.sidecar_path(state.stem)
            entry = {"image": state.path, "json": json_path, "sidecar": sidecar, "json_copy": "",
                     "sidecar_copy": "", "print": journal["_prints"].get(state.stem),
                     "kind": "edited" if state.edited else "other" if state.users else "agent",
                     "signature": _signature(json_path)}
            folder = os.path.join(store.agent_dir, "backups")
            if entry["signature"] is not None:
                entry["json_copy"] = _copy_to(folder, json_path, os.path.basename(json_path), stamp)
                if entry["kind"] != "agent":
                    guard.users_copies.append(entry["json_copy"])
            if os.path.exists(sidecar):
                entry["sidecar_copy"] = _copy_to(folder, sidecar, os.path.basename(sidecar), stamp)
            guard.entries[_key(state.path)] = entry
            journal["pages"][state.stem] = entry
        for work_dir, journal in journals.items():
            journal.pop("_prints", None)
            write_json_atomic(_journal_path(guard.stores[work_dir]), journal)
        return guard

    def preprocess_failures(self, results: Optional[Sequence[Any]] = None) -> Dict[str, str]:
        """Pages the preprocess did not write again → why. Their JSON is the one from before the
        run (or there is none), so the agent must not work on it."""
        from ..utils.path_manager import find_json_path

        contexts = {}
        for ctx in results or []:
            name = getattr(ctx, "image_name", None)
            if isinstance(name, str) and name:
                contexts[_key(name)] = ctx
        failures = {}
        for path in self.paths:
            entry = self.entries.get(_key(path)) or {}
            ctx = contexts.get(_key(path))
            reason = ""
            if ctx is not None:
                if getattr(ctx, "skipped", False):
                    reason = str(getattr(ctx, "skip_message", "") or getattr(ctx, "skip_reason", "") or "skipped")
                elif getattr(ctx, "success", None) is False or getattr(ctx, "translation_error", None):
                    reason = str(getattr(ctx, "translation_error", None) or getattr(ctx, "error", None) or "failed")
            if not reason:
                json_path = find_json_path(path)
                if not json_path or not os.path.exists(json_path):
                    reason = "no JSON was written"
                elif entry.get("signature") is not None and _signature(json_path) == entry["signature"]:
                    reason = "the JSON was not written again"
            if reason:
                failures[path] = reason
        return failures

    def page_finished(self, image_path: str) -> None:
        """The agent finished the page: it keeps this run's result from now on."""
        entry = self.entries.get(_key(image_path))
        if entry is None:
            return
        store = _store(image_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        done = self.finished.setdefault(store.work_dir, set())
        if stem in done:
            return
        done.add(stem)
        journal_path = _journal_path(store)
        journal = _read_json(journal_path, None)
        if isinstance(journal, dict) and stem in (journal.get("pages") or {}):
            journal["pages"].pop(stem)
            try:
                write_json_atomic(journal_path, journal)
            except OSError:
                logger.warning(f"agent: could not update {journal_path}", exc_info=True)

    def finish(self, delete_txt: Optional[Callable[[str], Any]] = None) -> List[str]:
        """End of the run (also when it stopped or failed): the unfinished pages get their earlier
        JSON and sidecar back, the finished ones are fingerprinted. Returns the restored pages."""
        restored: List[str] = []
        finished_paths = []
        for work_dir, store in self.stores.items():
            done = self.finished.get(work_dir, set())
            entries = {}
            for entry in self.entries.values():
                if _store(entry["image"]).work_dir == work_dir:
                    stem = os.path.splitext(os.path.basename(entry["image"]))[0]
                    entries[stem] = entry
                    if stem in done:
                        finished_paths.append(entry["image"])
            restored += _close(store, entries, done)
        if finished_paths:
            record_saved(finished_paths)
        if delete_txt is not None:  # the preprocess exported the original text of every page
            finished = {_key(p) for p in finished_paths}
            for path in self.paths:
                if _key(path) not in finished:
                    try:
                        delete_txt(path)
                    except Exception:
                        logger.warning(f"agent: could not remove the text export of {path}", exc_info=True)
        return restored
