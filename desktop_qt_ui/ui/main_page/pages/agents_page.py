"""Agents page: what the AI agent is doing, live, or replayed from a run log.

Layout (design doc, "Agents 頁面"), M2 skeleton:
* header: run status, current page, context usage bar, open a run log, problems-only filter
* left: pages with their state and review-flag count (click to filter the timeline)
* middle: activity timeline (tool calls with the agent's reason, results, flags, errors)
* right: live preview of one page, original -> inpainted -> result (``agent_preview``)
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

from .agent_preview import PagePreview
from .agent_results import flag_text, result_text

TOOL_KEYS = {
    "get_candidates": "Agent Tool get_candidates",
    "zoom": "Agent Tool zoom",
    "get_neighbors": "Agent Tool get_neighbors",
    "local_ocr": "Agent Tool local_ocr",
    "measure_bubble": "Agent Tool measure_bubble",
    "add_region": "Agent Tool add_region",
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
    "view_region": "Agent Tool view_region",
    "mask_select": "Agent Tool mask_select",
    "adjust_mask": "Agent Tool adjust_mask",
    "mask_paint": "Agent Tool mask_paint",
    "protect_lines": "Agent Tool protect_lines",
    "inpaint_region": "Agent Tool inpaint_region",
    "pick_candidate": "Agent Tool pick_candidate",
    "finish_review": "Agent Tool finish_review",
    "view_page": "Agent Tool view_page",
    "revise_text": "Agent Tool revise_text",
    "finish_proofread": "Agent Tool finish_proofread",
}
PAGE_STATE_KEYS = {
    "waiting": "Agent Page Waiting",
    "working": "Agent Page Working",
    "inpainting": "Agent Page Inpainting",
    "done": "Agent Page Done",
    "incomplete": "Agent Page Incomplete",
    "error": "Agent Page Error",
    "stopped": "Agent Page Stopped",
}
WARN = QColor(214, 140, 0)
ERROR = QColor(210, 50, 50)
MUTED = QColor(130, 130, 130)
ALL_PAGES = "__all__"


def _region_id(value) -> Optional[int]:
    """The region number of a tool call or flag (``3`` or ``"#3"``), for the preview highlight."""
    try:
        return int(str(value).lstrip("#")) if value not in (None, "", 0, "0") else None
    except (TypeError, ValueError):
        return None


def _own_region(event: dict, value) -> Optional[int]:
    """Only the translating agent's region numbers match the preview's boxes (the inpaint review,
    the proofreader and the font check number regions their own way)."""
    return _region_id(value) if (event.get("agent") or "translator") == "translator" else None


def _requests(event: dict) -> int:
    usage = event.get("usage")
    try:
        return int(usage.get("requests") or 0) if isinstance(usage, dict) else 0
    except (TypeError, ValueError):
        return 0


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
        self._translating = False
        self._proofreading = False
        self._finished: Optional[dict] = None
        """``done`` and ``pages`` of the translator's run_end; the status is built from it and the totals."""
        self._requests = 0
        """Model requests of every agent: translator and recall, inpaint review, proofreading."""
        self._rendering: Dict[str, int] = {}
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
        self.page_list.setMinimumWidth(200)
        self.page_list.currentItemChanged.connect(self._on_page_selected)
        self.timeline = TreeWidget(splitter)
        self.timeline.setColumnCount(5)
        self.timeline.setRootIsDecorated(False)
        self.timeline.setUniformRowHeights(True)
        self.timeline.setWordWrap(False)
        for column, width in enumerate((96, 56, 150, 280)):
            self.timeline.setColumnWidth(column, width)
        self.timeline.currentItemChanged.connect(self._on_row_selected)
        self.preview = PagePreview(self._t, splitter)
        splitter.addWidget(self.page_list)
        splitter.addWidget(self.timeline)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 560, 420])
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
        self.preview.refresh_texts()

    # --- events -------------------------------------------------------------
    def _on_agent_event(self, event: dict) -> None:  # worker thread
        self.event_received.emit(dict(event))

    def handle_event(self, event: dict) -> None:
        kind = event.get("type")
        page = event.get("page") or ""
        if kind in ("page_start", "page_done", "inpaint_done", "render_done") or event.get("boxes") is not None:
            self.preview.update_page(page, image=event.get("image"), boxes=event.get("boxes"),
                                     inpainted=event.get("inpainted"), output=event.get("output"))
        if kind == "run_start":
            self._reset([os.path.splitext(p)[0] for p in event.get("pages", [])])
            self._translating = True
            self._proofreading = False
            self._finished = None
            self._requests = 0
            self.status_label.setText(self._t("Agents Running", model=event.get("model") or "-"))
            self._update_memory(event.get("memory"))
        elif kind == "page_start":
            self.preview.follow(page)
            self._set_page(page, state="working")
            self.status_label.setText(self._t("Agents Page Progress", current=int(event.get("index", 0)) + 1,
                                              total=event.get("total", len(self._order)), page=page))
        elif kind == "tool_call":
            tool = event.get("tool", "")
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            ids = args.get("ids") if isinstance(args.get("ids"), list) and args.get("ids") else [None]
            row = self._add_row(event, self._t(TOOL_KEYS.get(tool, tool)) if tool in TOOL_KEYS else tool,
                                "", event.get("reason", ""), region=_own_region(event, args.get("id", ids[0])))
            row.setToolTip(2, json.dumps(event.get("args", {}), ensure_ascii=False)[:2000])
            self._pending_rows.setdefault(f"{page}|{tool}", []).append(row)
        elif kind == "tool_result":
            queue = self._pending_rows.get(f"{page}|{event.get('tool', '')}")
            row = queue.pop(0) if queue else None
            if row is not None:
                text, tip = result_text(self._t, event.get("tool", ""), event.get("summary", ""),
                                        event.get("ok", True), TOOL_KEYS)
                row.setText(3, text)
                row.setToolTip(3, tip)
                if not event.get("ok", True):
                    row.setData(0, Qt.ItemDataRole.UserRole + 1, True)
                    self._paint(row, WARN)
                self._apply_filter_to(row)
        elif kind == "flag":
            region = event.get("region")
            detail = flag_text(self._t, region, event.get("flag", ""))
            pending = self._pending_rows.get(f"{page}|flag_for_review")
            if pending:  # the flag_for_review call already has a row: highlight it instead of adding a duplicate
                row = pending[0]
                row.setText(3, detail)
                row.setData(0, Qt.ItemDataRole.UserRole + 1, True)
                self._apply_filter_to(row)
            else:
                row = self._add_row(event, self._t("Agents Flag"), detail, event.get("reason", ""), problem=True,
                                    region=_own_region(event, region))
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
        elif kind == "recall":
            if event.get("error"):
                row = self._add_row(event, self._t("Agents Recall"), str(event["error"]), "", problem=True)
                self._paint(row, ERROR)
            else:
                added = event.get("added") or []
                ignored = sum((event.get("ignored") or {}).values())
                detail = self._t("Agents Recall Detail", added=len(added), ignored=ignored)
                if added:
                    detail += " · " + ", ".join(f"#{i}" for i in added)
                self._add_row(event, self._t("Agents Recall"), detail, "")
        elif kind == "inpaint_start":
            self._set_page(page, state="inpainting")
            if not (self._translating or self._proofreading):  # the status line follows the main work
                self.status_label.setText(self._t("Agents Inpaint Progress", page=page))
            self._add_row(event, self._t("Agents Inpaint"),
                          self._t("Agents Inpaint Start", units=event.get("units", 0), issues=event.get("issues", 0)),
                          "")
        elif kind == "inpaint_done":
            self._requests += _requests(event)
            flagged = int(event.get("flagged") or 0)
            filled = int(event.get("filled") or 0)
            counts = {"units": event.get("units", 0), "retried": event.get("retried", 0), "flagged": flagged}
            detail = self._t("Agents Inpaint Detail Filled", filled=filled, **counts) if filled else \
                self._t("Agents Inpaint Detail", **counts)
            row = self._add_row(event, self._t("Agents Inpaint"), detail, "", problem=flagged > 0)
            if flagged:
                self._paint(row, WARN)
            self._set_page(page, state=event.get("status") or "done", flags=event.get("flags"),
                           confidence=event.get("confidence"))
            if not self._translating and not self._rendering:
                self.preview.follow(page)
            self._show_finished_if_idle()
        elif kind == "proofread_start":
            self._proofreading = True
            self.status_label.setText(self._t("Agents Proofread Progress"))
            self._add_row(event, self._t("Agents Proofread"),
                          self._t("Agents Proofread Start", pages=event.get("pages", 0),
                                  regions=event.get("regions", 0)), "")
        elif kind == "proofread_done":
            self._requests += _requests(event)
            self._proofreading = False
            flagged = int(event.get("flagged") or 0)
            row = self._add_row(event, self._t("Agents Proofread"),
                                self._t("Agents Proofread Detail", revised=event.get("revised", 0), flagged=flagged),
                                event.get("note", ""), problem=flagged > 0)
            if flagged:
                self._paint(row, WARN)
            self._show_finished_if_idle()
        elif kind == "fonts":
            unfixed = int(event.get("unfixed") or 0)
            reasons = []
            for change in event.get("changes") or []:
                why = self._t("Agents Fonts Missing", chars=change["missing"]) if change.get("missing") else \
                    self._t("Agents Fonts Layout") if change.get("reason") == "layout" else ""
                reasons.append(f"#{change.get('region')} → {change.get('to')}" + (f" ({why})" if why else ""))
            row = self._add_row(event, self._t("Agents Fonts"),
                                self._t("Agents Fonts Detail", changed=event.get("changed", 0), unfixed=unfixed),
                                "; ".join(reasons), problem=unfixed > 0)
            if unfixed:
                self._paint(row, WARN)
        elif kind == "work_glossary":
            added = event.get("added") or []
            skipped = event.get("skipped") or {}
            reason = "; ".join(f"{t.get('source')} → {t.get('target')} ({t.get('category')})" for t in added)
            if skipped:
                left = self._t("Agents Work Glossary Skipped", known=skipped.get("known", 0),
                               other=skipped.get("other", 0), unused=skipped.get("unused", 0))
                reason = f"{reason}; {left}" if reason else left
            row = self._add_row(event, self._t("Agents Work Glossary"),
                                self._t("Agents Work Glossary Detail", added=len(added), file=event.get("file", "")),
                                reason)
            if not added:
                self._paint(row, MUTED)
        elif kind == "render_start":
            run = str(event.get("run") or "")
            self._rendering[run] = int(event.get("pages") or 0)
            text = self._t("Agents Render Start", pages=event.get("pages", 0))
            self.status_label.setText(text)
            self._add_row(event, self._t("Agents Render"), text, "")
        elif kind == "render_done":
            self.preview.follow(page)
            output = event.get("output") or ""
            row = self._add_row(event, self._t("Agents Render"),
                                self._t("Agents Render Detail", file=os.path.basename(output)), "")
            row.setToolTip(3, output)
            self._paint(row, MUTED)
            run = str(event.get("run") or "")
            left = self._rendering.get(run, 0) - 1
            if left > 0:
                self._rendering[run] = left
                self.status_label.setText(self._t("Agents Render Progress", page=page))
            else:
                self._rendering.pop(run, None)
                if self._finished is not None and not self._rendering:
                    self.status_label.setText(self._finished_status())
        elif kind == "error":
            row = self._add_row(event, self._t("Agents Error"), event.get("error", ""), "", problem=True)
            self._paint(row, ERROR)
            if event.get("agent") == "proofread":
                self._proofreading = False
                self._show_finished_if_idle()
            if event.get("agent") == "inpaint" and self._pages.get(page, {}).get("state") == "inpainting":
                # the review failed; the translation stands and render inpaints the page again
                self._set_page(page, state="done")
                self._show_finished_if_idle()
        elif kind == "page_done":
            self._set_page(page, state=event.get("status") or "done", flags=event.get("flags"),
                           confidence=event.get("confidence"))
            self._update_memory(event.get("memory"))
        elif kind == "run_end":
            totals = event.get("totals") or {}
            self._translating = False
            self._requests += int(totals.get("requests") or 0) + int(totals.get("recall_requests") or 0)
            self._finished = {"done": event.get("done", 0), "pages": event.get("pages", 0)}
            reviewing = [p for p in self._order if self._pages.get(p, {}).get("state") == "inpainting"]
            if reviewing:  # the inpaint review of the last page(s) is still running
                self.status_label.setText(self._t("Agents Inpaint Progress", page=reviewing[0]))
            else:
                self.status_label.setText(self._finished_status())
            self._update_memory(event.get("memory"))

    def _finished_status(self) -> str:
        """Pages done by the translator, flags as the page list counts them (every agent's), requests."""
        flagged = sum(int(info.get("flags") or 0) for info in self._pages.values())
        return self._t("Agents Finished", done=self._finished["done"], pages=self._finished["pages"],
                       flagged=flagged, requests=self._requests)

    def _show_finished_if_idle(self) -> None:
        if self._translating or self._proofreading or self._finished is None:
            return
        if not any(info.get("state") == "inpainting" for info in self._pages.values()):
            self.status_label.setText(self._finished_status())

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
        self._rendering.clear()
        self.preview.reset()
        self.preview.set_following(True)
        self._update_context(0)
        self._rebuild_page_list()

    def _add_row(self, event: dict, action: str, detail: str, reason: str, problem: bool = False,
                 region=None) -> QTreeWidgetItem:
        ts = str(event.get("ts", ""))
        clock = ts[11:19] if len(ts) >= 19 else ts
        row = QTreeWidgetItem([clock, event.get("page") or "", action, detail, reason or ""])
        row.setToolTip(3, detail)
        row.setToolTip(4, reason or "")
        row.setData(0, Qt.ItemDataRole.UserRole, event.get("page") or "")
        row.setData(0, Qt.ItemDataRole.UserRole + 1, problem)
        row.setData(0, Qt.ItemDataRole.UserRole + 2, _region_id(region))
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

    def _set_page(self, page: str, state: str, flags=None, confidence=None) -> None:
        if not page:
            return
        if page not in self._pages:
            self._order.append(page)
        info = self._pages.setdefault(page, {"state": state, "flags": 0})
        info["state"] = state
        if flags is not None:
            info["flags"] = int(flags)
        if confidence is not None:
            info["confidence"] = float(confidence)
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
                if info.get("confidence") is not None:
                    text += " · " + self._t("Agents Page Confidence", percent=round(info["confidence"] * 100))
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
        if self._page_filter == ALL_PAGES:
            self.preview.set_following(True)
        else:
            self.preview.show_page(self._page_filter)

    def _on_row_selected(self, current, _previous) -> None:
        """A timeline row: show its page, and its region when it has one."""
        if current is None:
            return
        page = current.data(0, Qt.ItemDataRole.UserRole) or ""
        if page and page in self.preview.pages:
            self.preview.show_page(page, highlight=current.data(0, Qt.ItemDataRole.UserRole + 2))

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
