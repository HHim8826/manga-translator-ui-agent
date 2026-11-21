"""Event log of an agent run: one JSON object per line, also pushed to live listeners.

Event types (design doc, "事件紀錄"): ``run_start`` (with ``chapter`` and ``batch``: the chapters
of one workflow run share the batch id), ``page_start``, ``tool_call``, ``tool_result``, ``flag``,
``usage``, ``compact``, ``recall``, ``error``, ``page_done``, ``run_end`` (``reason``: empty, or
``stopped``/``failed`` when the run ended early), ``inpaint_start``/``inpaint_done``,
``proofread_start``/``proofread_done``, ``fonts``, ``font_styles``, ``work_glossary`` (new terms
saved to the work's prompt file), ``render_start``/``render_done`` (the image paths and region boxes
the preview uses: "即時預覽") and ``stopped`` (the user stopped the run; ``stage`` is ``agent`` or
``render``).
Listeners (the Agents page) are called on the agent's thread and must not block.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional

Listener = Callable[[Dict[str, Any]], None]

_listeners: List[Listener] = []
_lock = threading.Lock()


def add_listener(listener: Listener) -> None:
    with _lock:
        if listener not in _listeners:
            _listeners.append(listener)


def remove_listener(listener: Listener) -> None:
    with _lock:
        if listener in _listeners:
            _listeners.remove(listener)


def new_run_id(now: Optional[_dt.datetime] = None) -> str:
    return (now or _dt.datetime.now()).strftime("r%Y%m%d-%H%M%S")


class EventLog:
    def __init__(self, path: str, run_id: str):
        self.path = path
        self.run_id = run_id
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, type: str, page: Optional[str] = None, agent: str = "translator", **fields: Any) -> Dict[str, Any]:
        event = {"ts": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                 "run": self.run_id, "agent": agent, "page": page, "type": type, **fields}
        if self._fh and not self._fh.closed:
            self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        with _lock:
            listeners = list(_listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass
        return event

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()


def read_events(path: str) -> List[Dict[str, Any]]:
    events = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events
