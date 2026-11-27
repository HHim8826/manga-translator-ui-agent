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
  (``agent/run_journal.json``, written once) lists them; each page the run finishes is added to
  ``agent/run_journal.done``, so the pages left are the journal's without those;
* a page the preprocess did not write again (it failed there) is not given to the agent: its JSON is
  the earlier one, maybe for another image or other boxes;
* when the run stops or fails, the pages it did not finish get their JSON and sidecar back, so a page
  the agent never reached keeps its earlier translation and status. A page whose files cannot be put
  back (a file in use) keeps its copies and its journal entry, and is put back later;
* a journal left by a run that was killed is restored before the next run (``recover``).

The journal names the run that owns the chapter (``owner``: process, computer, start time). Only
that run changes the chapter's pages: reading the state (``previous_states``) writes nothing, and a
journal is restored only once its run has ended (``owner_alive``); a second run on a chapter that
another one is working on does not start (``ChapterBusy``).

Copies of the user's own work (edited after a run, or translated outside the AI Agent) stay in
``backups``; the copies of the agent's own results are removed when the run ends.
"""
from __future__ import annotations

import errno
import hashlib
import logging
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from . import redo as redo_steps
from .store import AgentStore, _read_json, write_json_atomic
from .store import page_stem

logger = logging.getLogger("manga_translator.agent")

EDIT_MARGIN = 30.0
FINGERPRINTS = "saved_json.json"
USERS_SAVE = "saved in the editor during a run"
"""What ``FINGERPRINTS`` has for a JSON the user saved in the editor during a run: no fingerprint, so the
next run tells the page edited and keeps a copy of it (``record_saved``)."""
INPAINTED = "saved_inpainted.json"
"""Page stem -> [mtime_ns, size] of its inpainted image as the workflow left it: a redo does not
review again an inpainted image the user changed (their repairs are in it)."""
STATS = "saved_json.stat.json"
"""``[mtime_ns, size, fingerprint]`` of each page's JSON when its fingerprint was taken: the scan before
the rerun question does not read a JSON again that still has its time and size (each read the whole
file before: 100 pages of 1 MiB took about 0.34 s on the GUI thread)."""
JOURNAL = "run_journal.json"
LOCK = "run.lock"
"""Locked by the run that works on the chapter, from before it reads the chapter's state until it
ends (``_lock``): two runs that start at the same moment cannot both take it. The system lets go of
the lock when the process ends, also when it is killed."""

_ACTIVE: Set[str] = set()
"""Owner tokens of the runs of this process that have not ended (their guard has not finished)."""


class ChapterBusy(RuntimeError):
    """Another run is working on the chapter: a second one would change the same pages."""


class PagesNotRestored(ChapterBusy):
    """Pages of a run that was killed cannot be put back yet (a file is in use): a new run would
    replace the journal that keeps their copies."""


def _process_start(pid: int) -> Optional[float]:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _lock(store: AgentStore):
    """The chapter's ``LOCK``, held: an open file, or None when this file system cannot lock (the
    journal's owner is all that tells a run is working on the chapter then). ChapterBusy when another
    run holds it (also one of this process)."""
    os.makedirs(store.agent_dir, exist_ok=True)
    handle = open(os.path.join(store.agent_dir, LOCK), "a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK, getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
                         getattr(errno, "EDEADLOCK", errno.EDEADLK)}:
            raise ChapterBusy(f"another AI Agent run is working on {os.path.dirname(store.work_dir)}; "
                              f"wait for it to end, or stop it") from exc
        logger.warning(f"agent: cannot lock {store.agent_dir}; only its journal tells a run works on it",
                       exc_info=True)
        return None
    except BaseException:
        handle.close()
        raise
    return handle


def _unlock(handle) -> None:
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        logger.debug("agent: unlocking a chapter failed (closing it lets go too)", exc_info=True)
    finally:
        handle.close()


def _lock_free(store: AgentStore) -> bool:
    """No run holds the chapter's ``LOCK`` (it is taken and let go at once). False when one does, and
    when this file system cannot lock (the journal's owner is all that tells then)."""
    try:
        handle = _lock(store)
    except (ChapterBusy, OSError):
        return False
    if handle is None:
        return False
    _unlock(handle)
    return True


def _run_on(store: AgentStore, journal: Optional[dict]) -> bool:
    """A run is working on the chapter of ``journal``: its owner is alive (``owner_alive``) and holds the
    chapter's ``LOCK``. With the lock free, a journal of another computer (the folder was moved or
    synced) or of a process whose id the system gave to another one locked the chapter for good: it
    was shown as running and no run started (review 2026-11-28 B18)."""
    return bool(journal) and owner_alive(journal.get("owner")) and not _lock_free(store)


def _new_owner() -> dict:
    owner = {"pid": os.getpid(), "host": socket.gethostname(), "token": uuid.uuid4().hex,
             "process_started": _process_start(os.getpid())}
    _ACTIVE.add(owner["token"])
    return owner


def owner_alive(owner: Any) -> bool:
    """The run that wrote a journal is still going: one of this process that has not ended, or another
    process on this computer that is still running (the same process: the same id and start time). A
    journal of another computer counts as alive (its process cannot be checked); a journal without an
    owner was written before owners were kept, by a run that is over."""
    if not isinstance(owner, dict):
        return False
    if owner.get("host") and owner.get("host") != socket.gethostname():
        return True
    try:
        pid = int(owner.get("pid"))
    except (TypeError, ValueError):
        return False
    if pid == os.getpid():
        return owner.get("token") in _ACTIVE
    try:
        import psutil
    except ImportError:  # cannot be checked: like before owners were kept
        return False
    try:
        if not psutil.pid_exists(pid):
            return False
        started, recorded = _process_start(pid), owner.get("process_started")
        return started is None or recorded is None or abs(started - float(recorded)) < 1.0
    except Exception:
        return True  # cannot tell: leave the chapter alone


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
    busy: bool = False
    """Another run is working on the page's chapter (its journal's owner is alive)."""
    pending: Tuple[str, ...] = ()
    """Steps an earlier run could not do on the page (``redo.py``: ``inpaint_review``, ``proofread``)."""
    inpaint_edited: bool = False
    """The inpainted image changed after the workflow left it (the user repaired it). Only told for a
    page with an inpaint review to redo."""
    resets: str = ""
    """When the API's usage limit that stopped them resets (an ISO time), if it said and it is still
    to come (``redo.resets_of``)."""

    @property
    def stem(self) -> str:
        return page_stem(self.path)

    @property
    def users(self) -> bool:
        """The JSON holds the user's own work (also a page the user took in the editor during a run:
        status ``edited``)."""
        return self.edited or self.status == "edited" or (not self.status and self.translated)


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


def _stats_path(store: AgentStore) -> str:
    return os.path.join(store.agent_dir, STATS)


def _read_stats(store: AgentStore) -> dict:
    """``STATS``; empty when it is missing or broken (it only saves reading the JSON again)."""
    try:
        stats = _read_json(_stats_path(store), {})
    except (OSError, ValueError):
        return {}
    return stats if isinstance(stats, dict) else {}


def _stat(path: str) -> Optional[list]:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return [stat.st_mtime_ns, stat.st_size]


def _inpainted_changed(image_path: str, saved: Any, log_time: Optional[float]) -> bool:
    """The page's inpainted image is not the one the workflow left (``INPAINTED``); without a record
    (a run before them), it was written more than ``EDIT_MARGIN`` seconds after the latest run log."""
    from ..utils.path_manager import find_inpainted_path

    inpainted = find_inpainted_path(image_path)
    now = _stat(inpainted) if inpainted else None
    if now is None:
        return False
    if isinstance(saved, list) and len(saved) == 2:
        return list(saved) != now
    return log_time is not None and now[0] / 1e9 > log_time + EDIT_MARGIN


def _changed_since(path: str, saved: str, known: Any) -> bool:
    """The JSON's content is not the one fingerprinted ``saved``. Its time and size as they were when
    that fingerprint was taken (``known``, from ``STATS``) say it is without reading it."""
    if isinstance(known, list) and len(known) == 3 and known[2] == saved and known[:2] == _stat(path):
        return False
    return _fingerprint(path) != saved


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


def _done_path(store: AgentStore) -> str:
    """The pages of the journal the run finished, one name per line (``RunGuard.page_finished``)."""
    return os.path.join(store.agent_dir, JOURNAL[:-len(".json")] + ".done")


def _read_journal(store: AgentStore) -> Optional[dict]:
    """The chapter's journal with only the pages its run has not finished; None when there is none."""
    journal = _read_json(_journal_path(store), None) if os.path.isdir(store.agent_dir) else None
    if not isinstance(journal, dict):
        return None
    try:
        with open(_done_path(store), encoding="utf-8") as fh:
            done = {line.strip() for line in fh if line.strip()}
    except OSError:
        done = set()
    if done:
        journal["pages"] = {stem: e for stem, e in (journal.get("pages") or {}).items() if stem not in done}
    return journal


def previous_states(paths: Sequence[str]) -> List[PageState]:
    """Every page's state from the earlier runs of its chapter. Nothing is written: a page of a run
    that was killed is told as it will be once its journal is restored (from the copies), and the
    pages of a chapter another run is working on are ``busy``."""
    from ..utils.path_manager import find_json_path

    stores: Dict[str, AgentStore] = {}
    prints: Dict[str, dict] = {}
    stats: Dict[str, dict] = {}
    log_times: Dict[str, Optional[float]] = {}
    journals: Dict[str, Tuple[dict, bool]] = {}
    pending_steps: Dict[str, dict] = {}
    inpainted: Dict[str, dict] = {}
    states = []
    for path in paths:
        store = _store(path)
        key = store.work_dir
        if key not in stores:
            stores[key] = store
            prints[key] = _read_json(_fingerprints_path(store), {}) if os.path.isdir(store.agent_dir) else {}
            stats[key] = _read_stats(store) if prints[key] else {}
            latest = store.latest_run_log() if os.path.isdir(store.agent_dir) else None
            log_times[key] = os.path.getmtime(latest) if latest else None
            journal = _read_journal(store)
            journals[key] = ((journal.get("pages") or {}) if journal else {}, _run_on(store, journal))
            pending_steps[key] = redo_steps.read(store)
        stem = page_stem(path)
        pending, busy = journals[key]
        entry = pending.get(stem) if not busy and isinstance(pending.get(stem), dict) else None
        restorable = entry is not None and bool(entry.get("json_copy")) and os.path.exists(entry["json_copy"])
        side_path = store.sidecar_path(stem)
        json_path = find_json_path(path) or ""
        read_json = json_path
        saved = prints[key].get(stem)
        known = stats[key].get(stem)
        if restorable:  # what the restore of the killed run's journal will put back
            side_path = entry.get("sidecar_copy") or ""
            read_json = entry["json_copy"]
            saved = entry.get("print")
            known = None
        side = _read_json(side_path, None) if side_path and os.path.exists(side_path) else None
        status = str((side or {}).get("status") or "") if isinstance(side, dict) else ""
        edited = translated = False
        exists = bool(read_json) and os.path.exists(read_json)
        if status and exists:
            if saved:
                edited = _changed_since(read_json, saved, known)
            elif log_times[key] is not None:
                edited = os.path.getmtime(read_json) > log_times[key] + EDIT_MARGIN
        elif exists:
            translated = has_translation(read_json)
        steps = redo_steps.steps_of(pending_steps[key], stem) if status else ()
        inpaint_edited = False
        if redo_steps.INPAINT_REVIEW in steps:
            if key not in inpainted:
                inpainted[key] = _read_json(os.path.join(store.agent_dir, INPAINTED), {})
            inpaint_edited = _inpainted_changed(path, inpainted[key].get(stem), log_times[key])
        states.append(PageState(path, status, edited, json_path, translated, busy, steps, inpaint_edited,
                                redo_steps.resets_of(pending_steps[key], stem) if steps else ""))
    return states


def record_saved(paths: Sequence[str]) -> None:
    """Fingerprints of the pages' JSON as the workflow leaves them, and the time and size of their
    inpainted images (``INPAINTED``). A JSON the user saved in the editor during the run is not the
    workflow's: it gets ``USERS_SAVE``, which no file matches, so the next run tells it edited."""
    from ..utils.path_manager import find_inpainted_path, find_json_path
    from . import json_guard

    by_store: Dict[str, List[str]] = {}
    for path in paths:
        by_store.setdefault(_store(path).work_dir, []).append(path)
    for work_dir, pages in by_store.items():
        store = AgentStore(work_dir)
        target = _fingerprints_path(store)
        prints = _read_json(target, {})
        stats = _read_stats(store)
        inpainted_target = os.path.join(store.agent_dir, INPAINTED)
        inpainted = _read_json(inpainted_target, {})
        for path in pages:
            stem = page_stem(path)
            image = find_inpainted_path(path)
            signature = _stat(image) if image else None
            if signature is not None:
                inpainted[stem] = signature
            else:
                inpainted.pop(stem, None)
            json_path = find_json_path(path)
            if json_path and json_guard.taken(json_path):
                # the user's file, saved in the editor during the run: fingerprinted as the run's, the
                # next run took it for the agent's own and dropped its copy (review 2026-11-28 B3)
                prints[stem] = USERS_SAVE
                stats.pop(stem, None)
            elif json_path and os.path.exists(json_path):
                before = _stat(json_path)
                prints[stem] = _fingerprint(json_path)
                if before is not None and before == _stat(json_path):  # not written while it was read
                    stats[stem] = [*before, prints[stem]]
                else:
                    stats.pop(stem, None)
        for name, data in ((target, prints), (_stats_path(store), stats), (inpainted_target, inpainted)):
            try:
                write_json_atomic(name, data)
            except OSError:
                logger.warning(f"agent: could not save {name}", exc_info=True)


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
    """The page's JSON and sidecar (and on a check its inpainted image) as they were before the run;
    False when it had no JSON then (the run's partial work stays)."""
    if not entry.get("json_copy") or not os.path.exists(entry["json_copy"]):
        return False
    shutil.copy2(entry["json_copy"], entry["json"])
    sidecar = entry["sidecar"]
    if entry.get("sidecar_copy") and os.path.exists(entry["sidecar_copy"]):
        shutil.copy2(entry["sidecar_copy"], sidecar)
    elif os.path.exists(sidecar):  # made by this run
        os.remove(sidecar)
    if entry.get("inpainted_copy") and os.path.exists(entry["inpainted_copy"]):
        shutil.copy2(entry["inpainted_copy"], entry["inpainted"])
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


def _close(store: AgentStore, entries: Dict[str, dict], finished: Sequence[str],
           keep_current: bool = False) -> Tuple[List[str], List[str]]:
    """Put back the pages of a journal that were not finished, drop the agent's own copies and the
    journal; returns (the restored pages, the pages that could not be put back). A page the user
    saved in the editor during the run stays as they saved it; ``keep_current`` (a run that was
    killed: nobody knows who wrote what) keeps a copy of each JSON it replaces.

    A page whose files cannot be put back (one is in use) keeps its copies, and the journal stays
    with only its entries and no owner, so the next run puts it back (``recover``)."""
    from . import json_guard

    restored, failed, prints, steps = [], [], {}, {}
    for stem, entry in entries.items():
        if stem in finished or json_guard.taken(entry.get("json")):
            continue
        if keep_current and entry.get("json_copy") and os.path.exists(entry.get("json") or ""):
            try:
                _copy_to(os.path.join(store.agent_dir, "backups"), entry["json"],
                         os.path.basename(entry["json"]), time.strftime("%Y%m%d-%H%M%S") + ".interrupted")
            except OSError:
                logger.warning(f"agent: could not copy {entry['json']}", exc_info=True)
        try:
            if _restore(entry):
                restored.append(entry["image"])
                prints[stem] = entry.get("print")
                if isinstance(entry.get("pending"), dict):  # its steps to redo were for that JSON
                    steps[stem] = entry["pending"]
        except OSError:
            logger.exception(f"agent: could not restore {entry.get('json')}")
            failed.append(stem)
    if prints:
        _set_prints(store, prints)
    if steps:
        try:
            redo_steps.put_back(store, steps)
        except Exception:
            logger.warning("agent: could not put back the steps left to redo", exc_info=True)
    for stem, entry in entries.items():
        if stem in failed or entry.get("kind") == "revise":  # still needed / a check keeps every copy
            continue
        _remove(entry.get("sidecar_copy", ""))
        if entry.get("kind") == "agent":
            _remove(entry.get("json_copy", ""))
    if failed:
        try:
            write_json_atomic(_journal_path(store), {"started": time.strftime("%Y%m%d-%H%M%S"),
                                                     "pages": {stem: entries[stem] for stem in failed}})
        except OSError:
            logger.exception(f"agent: could not keep the journal of {store.work_dir}")
    else:
        _remove(_journal_path(store))
    _remove(_done_path(store))
    return restored, [entries[stem]["image"] for stem in failed]


def recover(paths: Sequence[str], held: Sequence[str] = ()) -> List[str]:
    """Restore the unfinished pages of a run that was killed (its journal is still there, and its
    run has ended: ``_run_on``); returns them. A chapter another run is working on is left; one of
    ``held`` (work dirs whose ``LOCK`` the caller holds) has none."""
    restored: List[str] = []
    seen = set()
    for path in paths:
        store = _store(path)
        if store.work_dir in seen:
            continue
        seen.add(store.work_dir)
        journal = _read_journal(store)
        if journal is None:
            continue
        if store.work_dir not in held and _run_on(store, journal):
            logger.warning(f"agent: another run is working on {store.work_dir}; its journal is left alone")
            continue
        entries = {stem: e for stem, e in (journal.get("pages") or {}).items() if isinstance(e, dict)}
        pages, failed = _close(store, entries, (), keep_current=True)
        if pages:
            logger.warning(f"agent: an earlier run ended without finishing {len(pages)} page(s); "
                           f"their translations from before it are back: {', '.join(map(os.path.basename, pages))}")
        if failed:
            logger.warning(f"agent: {len(failed)} page(s) of an earlier run still cannot be put back (a file is "
                           f"in use): {', '.join(map(os.path.basename, failed))}")
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
        self.owner: dict = {}
        """The run in the journals (``owner_alive``)."""
        self.not_restored: List[str] = []
        """Pages ``finish`` could not put back (a file in use): their copies and journal entries stay."""
        self.locks: Dict[str, Any] = {}
        """Work dir -> the chapter's ``LOCK``, held until ``finish``."""

    @classmethod
    def begin(cls, paths: Sequence[str], revise: bool = False) -> "RunGuard":
        """Copies and the journal before the preprocess, or before a check (``revise``: no preprocess;
        the inpainted images are copied too, and every copy stays in ``backups``)."""
        from ..utils.path_manager import find_inpainted_path, find_json_path, get_json_path

        guard = cls(paths)
        try:
            for path in paths:  # nothing is written while another run works on one of the chapters
                store = _store(path)
                if store.work_dir in guard.locks:
                    continue
                # taken before the journal is read: two runs that start at once cannot both see none
                guard.locks[store.work_dir] = _lock(store)
                journal = _read_journal(store)
                # with the lock held no other run is on the chapter: the journal's owner is asked only on
                # a file system that cannot lock (review 2026-11-28 B18)
                if journal is not None and guard.locks[store.work_dir] is None and owner_alive(journal.get("owner")):
                    raise ChapterBusy(f"another AI Agent run is working on {os.path.dirname(store.work_dir)}; "
                                      f"wait for it to end, or stop it")
            # a run that was killed: its pages are put back first
            recover(paths, held=[d for d, handle in guard.locks.items() if handle is not None])
            for work_dir in guard.locks:
                # pages recover could not put back (a file in use) keep their entries in the journal:
                # this run's journal would replace it and lose their copies (review 2026-11-28 B4)
                left = _read_journal(AgentStore(work_dir))
                if left is not None:
                    pages = ", ".join(sorted(str(stem) for stem in (left.get("pages") or {})))
                    raise PagesNotRestored(f"pages of an earlier AI Agent run on {os.path.dirname(work_dir)} cannot "
                                           f"be put back as they were, a file is in use ({pages}): close the "
                                           f"program that has them open and start again")
            guard.owner = _new_owner()
            guard._begin(paths, revise)
        except BaseException:
            _ACTIVE.discard(guard.owner.get("token"))
            guard.release()
            raise
        return guard

    def release(self) -> None:
        """Let go of the chapters' locks (``finish`` does; the system does when the process ends)."""
        locks, self.locks = self.locks, {}
        for handle in locks.values():
            _unlock(handle)

    def _begin(self, paths: Sequence[str], revise: bool) -> None:
        from ..utils.path_manager import find_inpainted_path, find_json_path, get_json_path

        guard = self
        states = previous_states(paths)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        journals: Dict[str, dict] = {}
        pending: Dict[str, dict] = {}
        for state in states:
            store = _store(state.path)
            guard.stores.setdefault(store.work_dir, store)
            journal = journals.setdefault(store.work_dir, {"started": stamp, "owner": guard.owner, "pages": {},
                                                           "_prints": _read_json(_fingerprints_path(store), {})})
            json_path = state.json_path or find_json_path(state.path) or get_json_path(state.path, create_dir=False)
            sidecar = store.sidecar_path(state.stem)
            entry = {"image": state.path, "json": json_path, "sidecar": sidecar, "json_copy": "",
                     "sidecar_copy": "", "print": journal["_prints"].get(state.stem),
                     "kind": "revise" if revise else "edited" if state.edited else "other" if state.users else "agent",
                     "signature": _signature(json_path)}
            folder = os.path.join(store.agent_dir, "backups")
            if not revise:  # the preprocess forgets the page's steps to redo; a page put back gets them back
                if store.work_dir not in pending:
                    pending[store.work_dir] = redo_steps.read(store)
                entry["pending"] = redo_steps.of_page(pending[store.work_dir], state.stem)
            if entry["signature"] is not None:
                entry["json_copy"] = _copy_to(folder, json_path, os.path.basename(json_path), stamp)
                if entry["kind"] != "agent":
                    guard.users_copies.append(entry["json_copy"])
            if os.path.exists(sidecar):
                entry["sidecar_copy"] = _copy_to(folder, sidecar, os.path.basename(sidecar), stamp)
            inpainted = find_inpainted_path(state.path) if revise else None
            if inpainted and os.path.exists(inpainted):
                entry["inpainted"] = inpainted
                entry["inpainted_copy"] = _copy_to(folder, inpainted, os.path.basename(inpainted), stamp)
            guard.entries[_key(state.path)] = entry
            journal["pages"][state.stem] = entry
        for work_dir, journal in journals.items():
            journal.pop("_prints", None)
            store = guard.stores[work_dir]
            _remove(_done_path(store))
            write_json_atomic(_journal_path(store), journal)

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
        stem = page_stem(image_path)
        done = self.finished.setdefault(store.work_dir, set())
        if stem in done:
            return
        done.add(stem)
        # one line per finished page: the journal itself is written once, at the start (rewriting it
        # for each page wrote it n times over)
        done_path = _done_path(store)
        try:
            with open(done_path, "a", encoding="utf-8") as fh:
                fh.write(stem + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            logger.warning(f"agent: could not update {done_path}", exc_info=True)

    def finish(self, delete_txt: Optional[Callable[[str], Any]] = None) -> List[str]:
        """End of the run (also when it stopped or failed): the unfinished pages get their earlier
        JSON and sidecar back, the finished ones are fingerprinted. Returns the restored pages; the
        pages that could not be put back are in ``not_restored``."""
        try:
            return self._finish(delete_txt)
        finally:
            self.release()

    def _finish(self, delete_txt: Optional[Callable[[str], Any]]) -> List[str]:
        restored: List[str] = []
        finished_paths = []
        try:
            for work_dir, store in self.stores.items():
                done = self.finished.get(work_dir, set())
                entries = {}
                for entry in self.entries.values():
                    if _store(entry["image"]).work_dir == work_dir:
                        stem = page_stem(entry["image"])
                        entries[stem] = entry
                        if stem in done:
                            finished_paths.append(entry["image"])
                pages, failed = _close(store, entries, done)
                restored += pages
                self.not_restored += failed
        finally:
            _ACTIVE.discard(self.owner.get("token"))
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
