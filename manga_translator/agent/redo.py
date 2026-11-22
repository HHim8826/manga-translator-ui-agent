"""Steps an AI Agent run could not do on a page, so the user can do them again by hand ("redo").

The inpaint review of a page and the proofreading of a chapter need the API after the translation is
done. When one fails (most often the API's usage limit: an API proxy answers 429
``usage_limit_reached`` with the time it resets), the run goes on as before: the page renders with
the pipeline's inpainting, the chapter without proofreading. The step is written down here
(``agent/pending_steps.json``) with its error and, for a usage limit, the time the limit resets.

When the user starts the AI Agent on the chapter again, the rerun question offers "Redo the steps
that were not done" for those pages (``agent_rerun.py``). That run (``run_agent_workflow(redo=True)``)
does only those steps: no preprocess, no translation. The pages keep their text and the user's
changes (the proofreader leaves the boxes the user changed, as in a check), and the pages are
rendered again. A page the user changed in the editor after the run does not get its inpaint review
again: its inpainted image may hold the user's repairs.

A step is taken off the list when it is done, and a page's steps go when a new run translates the
page again (its preprocess replaces what the steps were for).
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict, Iterable, Tuple

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
    """step -> page stem -> {"error", "resets" (HH:MM, when a usage limit said), "at"}."""
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
    """``step`` could not be done on these pages."""
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


def forget(store: AgentStore, stems: Iterable[str]) -> None:
    """A new run translates these pages again: what was left to do on them is gone with their text."""
    for step in STEPS:
        done(store, step, stems)


def steps_of(data: Dict[str, Dict[str, dict]], stem: str) -> Tuple[str, ...]:
    return tuple(step for step in STEPS if stem in data.get(step, {}))


def resets_of(data: Dict[str, Dict[str, dict]], stem: str) -> str:
    """The latest time a usage limit said it resets, for the page's steps ("" when none said)."""
    times = [str(data[step][stem].get("resets") or "") for step in steps_of(data, stem)]
    return max((t for t in times if t), default="")


def summary(data: Dict[str, Any]) -> Dict[str, int]:
    return {step: len(data.get(step) or {}) for step in STEPS if data.get(step)}
