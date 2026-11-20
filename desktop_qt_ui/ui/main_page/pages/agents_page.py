"""Agents page: what the AI agent is doing, live, or replayed from a run log.

Layout (design doc, "Agents 頁面"), M2 skeleton:
* header: run status, current page, context usage bar, open a run log, problems-only filter
* left: pages with their state and review-flag count (click to filter the timeline)
* right: activity timeline (tool calls with the agent's reason, results, flags, errors)
* footer: memory counts (glossary, characters, story summary)

Events come from ``manga_translator.agent.events`` on the worker thread and are
moved to the UI thread through a Qt signal.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QListWidgetItem, QSplitter, QTreeWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    CheckBox,
    ListWidget,
    ProgressBar,
    PushButton,
    SubtitleLabel,
    TreeWidget,
)

from manga_translator.agent import events as agent_events

TOOL_KEYS = {
    "get_candidates": "Agent Tool get_candidates",
    "zoom": "Agent Tool zoom",
    "get_neighbors": "Agent Tool get_neighbors",
    "local_ocr": "Agent Tool local_ocr",
    "measure_bubble": "Agent Tool measure_bubble",
    "merge_regions": "Agent Tool merge_regions",
    "skip_region": "Agent Tool skip_region",
    "set_reading_order": "Agent Tool set_reading_order",
    "link_regions": "Agent Tool link_regions",
    "set_text": "Agent Tool set_text",
    "set_line_breaks": "Agent Tool set_line_breaks",
    "glossary_search": "Agent Tool glossary_search",
    "glossary_upsert": "Agent Tool glossary_upsert",
    "character_get": "Agent Tool character_get",
    "character_upsert": "Agent Tool character_upsert",
    "story_summary_get": "Agent Tool story_summary_get",
    "story_summary_update": "Agent Tool story_summary_update",
    "search_past_translations": "Agent Tool search_past_translations",
    "flag_for_review": "Agent Tool flag_for_review",
    "mark_page_done": "Agent Tool mark_page_done",
    "budget_status": "Agent Tool budget_status",
    "undo": "Agent Tool undo",
}
PAGE_STATE_KEYS = {
    "waiting": "Agent Page Waiting",
    "working": "Agent Page Working",
    "done": "Agent Page Done",
    "incomplete": "Agent Page Incomplete",
    "error": "Agent Page Error",
    "stopped": "Agent Page Stopped",
}
WARN = QColor(214, 140, 0)
ERROR = QColor(210, 50, 50)
MUTED = QColor(130, 130, 130)
ALL_PAGES = "__all__"


class AgentsPage(QWidget):
    event_received = pyqtSignal(dict)

    def __init__(self, translate: Callable[..., str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._t = translate
        self._pages: Dict[str, dict] = {}
        self._order: List[str] = []
        self._pending_rows: Dict[str, List[QTreeWidgetItem]] = {}
        self._page_filter = ALL_PAGES
        self._limit = 400_000
        self._build()
        self.event_received.connect(self.handle_event, Qt.ConnectionType.QueuedConnection)
        agent_events.add_listener(self._on_agent_event)
        self.destroyed.connect(lambda *_: agent_events.remove_listener(self._on_agent_event))
        self.refresh_texts()

    # --- layout ---------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(12)

        header = CardWidget(self)
        head = QVBoxLayout(header)
        head.setContentsMargins(16, 12, 16, 12)
        head.setSpacing(8)
        self.title_label = SubtitleLabel(header)
        self.subtitle_label = CaptionLabel(header)
        self.subtitle_label.setWordWrap(True)
        head.addWidget(self.title_label)
        head.addWidget(self.subtitle_label)

        row = QHBoxLayout()
        row.setSpacing(12)
        self.status_label = BodyLabel(header)
        self.context_label = CaptionLabel(header)
        self.context_bar = ProgressBar(header)
        self.context_bar.setRange(0, 1000)
        self.context_bar.setValue(0)
        self.context_bar.setFixedWidth(180)
        self.problems_only = CheckBox(header)
        self.problems_only.stateChanged.connect(lambda *_: self._apply_filter())
        self.open_log_button = PushButton(header)
        self.open_log_button.clicked.connect(self._open_run_log)
        row.addWidget(self.status_label, 1)
        row.addWidget(self.context_label)
        row.addWidget(self.context_bar)
        row.addWidget(self.problems_only)
        row.addWidget(self.open_log_button)
        head.addLayout(row)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.page_list = ListWidget(splitter)
        self.page_list.setMinimumWidth(170)
        self.page_list.currentItemChanged.connect(self._on_page_selected)
        self.timeline = TreeWidget(splitter)
        self.timeline.setColumnCount(5)
        self.timeline.setRootIsDecorated(False)
        self.timeline.setUniformRowHeights(True)
        self.timeline.setWordWrap(False)
        for column, width in enumerate((96, 56, 150, 280)):
            self.timeline.setColumnWidth(column, width)
        splitter.addWidget(self.page_list)
        splitter.addWidget(self.timeline)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([200, 800])
        root.addWidget(splitter, 1)

        self.memory_label = CaptionLabel(self)
        root.addWidget(self.memory_label)

    def refresh_texts(self) -> None:
        self.title_label.setText(self._t("Agents"))
        self.subtitle_label.setText(self._t("Agents Page Subtitle"))
        self.problems_only.setText(self._t("Agents Problems Only"))
        self.open_log_button.setText(self._t("Agents Open Run Log"))
        self.timeline.setHeaderLabels([self._t("Agents Column Time"), self._t("Agents Column Page"),
                                       self._t("Agents Column Action"), self._t("Agents Column Result"),
                                       self._t("Agents Column Reason")])
        if not self._order:
            self.status_label.setText(self._t("Agents Idle"))
            self.memory_label.setText("")
        self._update_context(None)
        self._rebuild_page_list()

    # --- events -------------------------------------------------------------
    def _on_agent_event(self, event: dict) -> None:  # worker thread
        self.event_received.emit(dict(event))

    def handle_event(self, event: dict) -> None:
        kind = event.get("type")
        page = event.get("page") or ""
        if kind == "run_start":
            self._reset([os.path.splitext(p)[0] for p in event.get("pages", [])])
            self.status_label.setText(self._t("Agents Running", model=event.get("model") or "-"))
            self._update_memory(event.get("memory"))
        elif kind == "page_start":
            self._set_page(page, state="working")
            self.status_label.setText(self._t("Agents Page Progress", current=int(event.get("index", 0)) + 1,
                                              total=event.get("total", len(self._order)), page=page))
        elif kind == "tool_call":
            tool = event.get("tool", "")
            row = self._add_row(event, self._t(TOOL_KEYS.get(tool, tool)) if tool in TOOL_KEYS else tool,
                                "", event.get("reason", ""))
            row.setToolTip(2, json.dumps(event.get("args", {}), ensure_ascii=False)[:2000])
            self._pending_rows.setdefault(f"{page}|{tool}", []).append(row)
        elif kind == "tool_result":
            queue = self._pending_rows.get(f"{page}|{event.get('tool', '')}")
            row = queue.pop(0) if queue else None
            if row is not None:
                row.setText(3, event.get("summary", ""))
                row.setToolTip(3, event.get("summary", ""))
                if not event.get("ok", True):
                    row.setData(0, Qt.ItemDataRole.UserRole + 1, True)
                    self._paint(row, WARN)
                self._apply_filter_to(row)
        elif kind == "flag":
            region = event.get("region")
            detail = (f"#{region} " if region else "") + str(event.get("flag", ""))
            pending = self._pending_rows.get(f"{page}|flag_for_review")
            if pending:  # the flag_for_review call already has a row: highlight it instead of adding a duplicate
                row = pending[0]
                row.setText(3, detail)
                row.setData(0, Qt.ItemDataRole.UserRole + 1, True)
                self._apply_filter_to(row)
            else:
                row = self._add_row(event, self._t("Agents Flag"), detail, event.get("reason", ""), problem=True)
            self._paint(row, WARN)
            info = self._pages.setdefault(page, {"state": "working", "flags": 0})
            info["flags"] = info.get("flags", 0) + 1
            self._rebuild_page_list()
        elif kind == "usage":
            self._limit = int(event.get("limit") or self._limit)
            self._update_context(int(event.get("prompt_tokens") or 0))
        elif kind == "compact":
            row = self._add_row(event, self._t("Agents Compact"),
                                self._t("Agents Compact Detail", tokens=event.get("tokens_before", 0)), "")
            self._paint(row, MUTED)
        elif kind == "error":
            row = self._add_row(event, self._t("Agents Error"), event.get("error", ""), "", problem=True)
            self._paint(row, ERROR)
        elif kind == "page_done":
            self._set_page(page, state=event.get("status") or "done", flags=event.get("flags"))
            self._update_memory(event.get("memory"))
        elif kind == "run_end":
            totals = event.get("totals") or {}
            self.status_label.setText(self._t("Agents Finished", done=event.get("done", 0),
                                              pages=event.get("pages", 0), flagged=event.get("flagged", 0),
                                              requests=totals.get("requests", 0)))
            self._update_memory(event.get("memory"))

    def replay(self, events: List[dict]) -> None:
        for event in events:
            self.handle_event(event)

    # --- helpers --------------------------------------------------------------
    def _reset(self, pages: List[str]) -> None:
        self.timeline.clear()
        self._pending_rows.clear()
        self._pages = {p: {"state": "waiting", "flags": 0} for p in pages}
        self._order = list(pages)
        self._page_filter = ALL_PAGES
        self._update_context(0)
        self._rebuild_page_list()

    def _add_row(self, event: dict, action: str, detail: str, reason: str, problem: bool = False) -> QTreeWidgetItem:
        ts = str(event.get("ts", ""))
        clock = ts[11:19] if len(ts) >= 19 else ts
        row = QTreeWidgetItem([clock, event.get("page") or "", action, detail, reason or ""])
        row.setToolTip(3, detail)
        row.setToolTip(4, reason or "")
        row.setData(0, Qt.ItemDataRole.UserRole, event.get("page") or "")
        row.setData(0, Qt.ItemDataRole.UserRole + 1, problem)
        self.timeline.addTopLevelItem(row)
        self._apply_filter_to(row)
        if not row.isHidden():
            self.timeline.scrollToItem(row)
        return row

    @staticmethod
    def _paint(row: QTreeWidgetItem, color: QColor) -> None:
        for column in range(row.columnCount()):
            row.setForeground(column, QBrush(color))

    def _apply_filter_to(self, row: QTreeWidgetItem) -> None:
        page_ok = self._page_filter == ALL_PAGES or row.data(0, Qt.ItemDataRole.UserRole) == self._page_filter
        problem_ok = not self.problems_only.isChecked() or bool(row.data(0, Qt.ItemDataRole.UserRole + 1))
        row.setHidden(not (page_ok and problem_ok))

    def _apply_filter(self) -> None:
        for index in range(self.timeline.topLevelItemCount()):
            self._apply_filter_to(self.timeline.topLevelItem(index))

    def _set_page(self, page: str, state: str, flags=None) -> None:
        if not page:
            return
        if page not in self._pages:
            self._order.append(page)
        info = self._pages.setdefault(page, {"state": state, "flags": 0})
        info["state"] = state
        if flags is not None:
            info["flags"] = int(flags)
        self._rebuild_page_list()

    def _rebuild_page_list(self) -> None:
        self.page_list.blockSignals(True)
        try:
            self.page_list.clear()
            everything = QListWidgetItem(self._t("Agents All Pages"))
            everything.setData(Qt.ItemDataRole.UserRole, ALL_PAGES)
            self.page_list.addItem(everything)
            for page in self._order:
                info = self._pages.get(page, {})
                state = self._t(PAGE_STATE_KEYS.get(info.get("state", "waiting"), "Agent Page Waiting"))
                text = f"{page} · {state}"
                if info.get("flags"):
                    text += " · " + self._t("Agents Flag Count", count=info["flags"])
                item = QListWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, page)
                if info.get("state") in ("error", "incomplete", "stopped"):
                    item.setForeground(QBrush(ERROR if info.get("state") == "error" else WARN))
                self.page_list.addItem(item)
                if page == self._page_filter:
                    self.page_list.setCurrentItem(item)
            if self._page_filter == ALL_PAGES:
                self.page_list.setCurrentItem(everything)
        finally:
            self.page_list.blockSignals(False)

    def _on_page_selected(self, current, _previous) -> None:
        if current is None:
            return
        self._page_filter = current.data(Qt.ItemDataRole.UserRole) or ALL_PAGES
        self._apply_filter()

    def _update_context(self, tokens: Optional[int]) -> None:
        if tokens is not None:
            self._last_tokens = tokens
        tokens = getattr(self, "_last_tokens", 0)
        self.context_bar.setValue(int(1000 * min(1.0, tokens / max(1, self._limit))))
        self.context_label.setText(self._t("Agents Context", used=f"{tokens / 1000:.0f}k",
                                           limit=f"{self._limit / 1000:.0f}k"))

    def _update_memory(self, counts: Optional[dict]) -> None:
        if not counts:
            return
        self.memory_label.setText(self._t("Agents Memory", glossary=counts.get("glossary", 0),
                                          characters=counts.get("characters", 0),
                                          story=counts.get("story_chars", 0)))

    def _open_run_log(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, self._t("Agents Open Run Log"), "", "Agent run log (*.jsonl)")
        if not path:
            return
        try:
            events = agent_events.read_events(path)
        except OSError as exc:
            self.status_label.setText(str(exc))
            return
        self.replay(events)
