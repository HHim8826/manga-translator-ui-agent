"""Steps an AI Agent run could not do on a page, so the user can do them again by hand ("redo").

The inpaint review of a page and the proofreading of a chapter need the API after the translation is
done. When one fails (most often the API's usage limit: an API proxy answers 429
``usage_limit_reached`` with the time it resets), the run goes on as before: the page renders with
the pipeline's inpainting, the chapter without proofreading. The step is written down here
(``agent/pending_steps.json``) with its error and, for a usage limit, the time the limit resets
(with its date and time zone: a limit hit before midnight may reset after it).

When the user starts the AI Agent on the chapter again, the rerun question offers "Redo the steps
that were not done" for those pages (``agent_rerun.py``). That run (``run_agent_workflow(redo=True)``)
does only those steps: no preprocess, no translation. The pages keep their text and the user's
changes (the proofreader leaves the boxes the user changed, as in a check), and the pages are
rendered again. A page the user changed in the editor after the run does not get its inpaint review
again: its inpainted image may hold the user's repairs.

A step is taken off the list when it is done, and a page's steps go when a new run translates the
page again (its preprocess replaces what the steps were for). When that run does not finish the page
and puts its JSON back, its steps come back with it (``put_back``, from the run's journal).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Any, Dict, Iterable, Optional, Tuple

from .store import AgentStore, _read_json, write_json_atomic

logger = logging.getLogger("manga_translator.agent")

PENDING = "pending_steps.json"
INPAINT_REVIEW = "inpaint_review"
PROOFREAD = "proofread"
STEPS = (INPAINT_REVIEW, PROOFREAD)
"""In the order a redo does them: the review may change a page's mask, the proofreading its text."""


def _path(store: AgentStore) -> str:
    return os.path.join(store.agent_dir, PENDING)


def read(store: AgentStore) -> Dict[str, Dict[str, dict]]:
    """step -> page stem -> {"error", "resets" (an ISO time, when a usage limit said; HH:MM in the
    first records), "at"}."""
    data = _read_json(_path(store), {}) if os.path.isdir(store.agent_dir) else {}
    if not isinstance(data, dict):
        return {}
    return {step: {str(stem): entry if isinstance(entry, dict) else {} for stem, entry in pages.items()}
            for step, pages in data.items() if step in STEPS and isinstance(pages, dict)}


def _write(store: AgentStore, data: Dict[str, Dict[str, dict]]) -> None:
    data = {step: pages for step, pages in data.items() if pages}
    path = _path(store)
    if not data:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("agent: could not remove %s", path, exc_info=True)
        return
    os.makedirs(store.agent_dir, exist_ok=True)
    write_json_atomic(path, data)


def failed(store: AgentStore, step: str, stems: Iterable[str], error: str = "", resets: str = "") -> None:
    """``step`` could not be done on these pages; ``resets``: when the usage limit that stopped it
    resets (``quota_resets``' ``resets_at``)."""
    stems = [str(s) for s in stems]
    if not stems:
        return
    data = read(store)
    at = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    entry = {"error": str(error or "")[:300], "at": at, **({"resets": resets} if resets else {})}
    data.setdefault(step, {}).update({stem: dict(entry) for stem in stems})
    _write(store, data)


def done(store: AgentStore, step: str, stems: Iterable[str]) -> None:
    """``step`` was done on these pages (or is no longer wanted there)."""
    stems = {str(s) for s in stems}
    data = read(store)
    if not stems or not any(stem in data.get(step, {}) for stem in stems):
        return
    data[step] = {stem: entry for stem, entry in data.get(step, {}).items() if stem not in stems}
    _write(store, data)


def of_page(data: Dict[str, Dict[str, dict]], stem: str) -> Dict[str, dict]:
    """step -> entry, for one page."""
    return {step: dict(data[step][stem]) for step in steps_of(data, stem)}


def put_back(store: AgentStore, pages: Dict[str, Dict[str, dict]]) -> None:
    """These pages are back as they were before a run (``resume``): so are their steps to redo
    (page stem -> step -> entry, as ``of_page`` gave them before the run)."""
    data = read(store)
    before = json.dumps(data, sort_keys=True)
    for stem, steps in pages.items():
        for step in STEPS:
            data.setdefault(step, {}).pop(stem, None)
            if isinstance(steps, dict) and isinstance(steps.get(step), dict):
                data[step][stem] = dict(steps[step])
    if json.dumps({k: v for k, v in data.items() if v}, sort_keys=True) != before:
        _write(store, data)


def forget(store: AgentStore, stems: Iterable[str]) -> None:
    """A new run translates these pages again: what was left to do on them is gone with their text."""
    for step in STEPS:
        done(store, step, stems)


def steps_of(data: Dict[str, Dict[str, dict]], stem: str) -> Tuple[str, ...]:
    return tuple(step for step in STEPS if stem in data.get(step, {}))


def _now() -> _dt.datetime:
    return _dt.datetime.now().astimezone()


def _time(value: Any) -> Optional[_dt.datetime]:
    try:
        when = _dt.datetime.fromisoformat(str(value or ""))
    except ValueError:
        return None
    return when if when.tzinfo is not None else when.astimezone()


def reset_time(entry: dict) -> Optional[_dt.datetime]:
    """When the usage limit of a step's entry resets; None when it did not say. The first records kept
    only HH:MM: that is the first such time after the entry was written (``at``)."""
    value = str(entry.get("resets") or "")
    when = _time(value) if len(value) > 5 else None
    if when is not None or not value:
        return when
    at = _time(entry.get("at"))
    try:
        hour, minute = (int(part) for part in value.split(":"))
        when = at.replace(hour=hour, minute=minute, second=0, microsecond=0) if at is not None else None
    except (ValueError, TypeError):
        return None
    return when if when is None or when >= at else when + _dt.timedelta(days=1)


def resets_of(data: Dict[str, Dict[str, dict]], stem: str, now: Optional[_dt.datetime] = None) -> str:
    """The latest time a usage limit said it resets for the page's steps, as an ISO time; "" when none
    said or that time has passed (the limit is back: a redo will not fail on it)."""
    times = [when for step in steps_of(data, stem) if (when := reset_time(data[step][stem])) is not None]
    latest = max(times, default=None)
    return latest.isoformat(timespec="minutes") if latest is not None and latest > (now or _now()) else ""


def latest_reset(values: Iterable[str], now: Optional[_dt.datetime] = None) -> str:
    """The latest of these ISO times (``resets_of``) still to come; ""."""
    times = [when for value in values if (when := _time(value)) is not None and when > (now or _now())]
    return max(times).isoformat(timespec="minutes") if times else ""


def reset_label(value: Any, now: Optional[_dt.datetime] = None) -> str:
    """An ISO time (or a datetime) as the user reads it: HH:MM today, MM/DD HH:MM on another day."""
    when = value if isinstance(value, _dt.datetime) else _time(value)
    if when is None:
        return str(value or "")
    when, now = when.astimezone(), (now or _now()).astimezone()
    return when.strftime("%H:%M" if when.date() == now.date() else "%m/%d %H:%M")


def summary(data: Dict[str, Any]) -> Dict[str, int]:
    return {step: len(data.get(step) or {}) for step in STEPS if data.get(step)}
