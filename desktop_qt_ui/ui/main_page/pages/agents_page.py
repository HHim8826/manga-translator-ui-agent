"""Agents page: what the AI agent is doing, live, or replayed from a run log.

Layout, M2 skeleton:
* header: run status, current page, context usage bar, open a run log, problems-only filter
* left: pages with their state and review-flag count (click to filter the timeline)
* middle: activity timeline (tool calls with the agent's reason, results, flags, errors)
* right: live preview of one page, original -> inpainted -> result (``agent_preview``)
* footer: memory counts (glossary, characters, story summary)

Events come from ``manga_translator.agent.events`` on the worker thread and are
moved to the UI thread through a Qt signal.

The chapters of one run (one ``batch`` id in ``run_start``) stay on the page together. Pages are
keyed by their file name; the second and later chapters prefix it with their run id
(``r20261121-152424|001``), so every chapter's ``001`` keeps its own row, state and preview.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QListWidgetItem, QSplitter, QTreeWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    CheckBox,
    FlowLayout,
    ListWidget,
    ProgressBar,
    PushButton,
    SubtitleLabel,
    TreeWidget,
)

from manga_translator.agent import events as agent_events

from ui.status_colors import ERROR, MUTED, WARN, follow_theme
from ui.status_colors import paint as paint_status

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
    "split_region": "Agent Tool split_region",
    "fit_region": "Agent Tool fit_region",
    "skip_region": "Agent Tool skip_region",
    "note_region": "Agent Tool note_region",
    "set_reading_order": "Agent Tool set_reading_order",
    "link_regions": "Agent Tool link_regions",
    "set_text": "Agent Tool set_text",
    "set_line_breaks": "Agent Tool set_line_breaks",
    "set_font_style": "Agent Tool set_font_style",
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
    "note_area": "Agent Tool note_area",
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
    "not_reached": "Agent Page Not Reached",
    "edited": "Agent Page Edited",
    "inpaint_failed": "Agent Page Inpaint Failed",
    "inpaint_unknown": "Agent Page Inpaint Unknown",
    "output_kept": "Agent Page Output Kept",
    "redo_failed": "Agent Page Redo Failed",
    "redo_none": "Agent Page Redo None",
    "redo_yours": "Agent Page Redo Yours",
}
HELD_STATES = ("inpaint_failed", "inpaint_unknown", "output_kept")
"""Pages a check finished but did not render again: their inpainted image is not up to their mask, or
their output image could not be backed up."""
REDO_STATES = ("redo_failed", "redo_none", "redo_yours")
"""Pages a redo (``manga_translator.agent.redo``) did not render again: their step failed again, or
nothing was left to do on them. They keep their output, like the held pages."""
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


_FLAG_TEXT = Qt.ItemDataRole.UserRole + 3
"""Set on a flag_for_review row whose result column the flag event wrote (numbered the agent's way, with
the flag's kind): its tool_result leaves it (review 2026-11-28 B19)."""


def _agent(event: dict) -> str:
    return str(event.get("agent") or "translator")


def _requests(event: dict) -> int:
    usage = event.get("usage")
    try:
        return int(usage.get("requests") or 0) if isinstance(usage, dict) else 0
    except (TypeError, ValueError):
        return 0


class AgentsPage(QWidget):
    event_received = pyqtSignal(dict)
    stop_requested = pyqtSignal()
    """The stop button: the main window stops the running task."""
    open_in_editor_requested = pyqtSignal(str)
    run_started = pyqtSignal()
    """An AI Agent run started (its ``workflow_stage`` ``start``: after the user chose what to redo)."""
    """An image path to open in the editor."""

    def __init__(self, translate: Callable[..., str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._t = translate
        self._pages: Dict[str, dict] = {}
        self._order: List[str] = []
        self._pending_rows: Dict[str, List[QTreeWidgetItem]] = {}
        self._replaying = False  # replay(): rows are added without following each one
        self._replay_reset = False
        self._page_filter = ALL_PAGES
        self._limit = 400_000
        self._translating = False
        self._proofreading = False
        self._finished: Optional[dict] = None
        """``done`` and ``pages`` of the translators' run_end, added over the chapters; the status is
        built from it and the totals."""
        self._finished_runs: Dict[str, dict] = {}
        self._requests = 0
        """Model requests of every agent: translator and recall, inpaint review, proofreading."""
        self._rendering: Dict[str, int] = {}
        self._unrendered = 0
        """Pages of the run the render left out: they have no agent translation."""
        self._render_skipped = 0
        """Pages the render did not save: an output file was there and overwriting is off ..."""
        self._render_failed = 0
        self._batch: Optional[str] = None
        self._chapters: Dict[str, str] = {}
        """Run id -> chapter folder name, in the order the chapters started."""
        self._prefix: Dict[str, str] = {}
        """Run id -> the prefix of its page keys (none for the first chapter)."""
        self._items: Dict[str, QListWidgetItem] = {}
        self._stopped = False
        self._failed = False
        self._task_running = False
        self._agent_task = False
        """The task running is an AI Agent run (it said so live): only that one stops from here."""
        self._stage = ""
        """What the run does before its first chapter starts (``workflow_stage``), for the status."""
        self._stopping = False
        self._viewing_log = ""
        """The run log the user opened; live events are kept meanwhile and not shown."""
        self._live_events: List[dict] = []
        """The live events since the current run started, for "back to the current run"."""
        self._live_batch: Optional[str] = None
        self._last_output = ""
        self._last_log_dir = ""
        self._build()
        follow_theme(self, self.timeline, self.page_list)  # the status colors of both themes
        self.preview.label_for = self._label
        self.event_received.connect(self._handle_live, Qt.ConnectionType.QueuedConnection)
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
        self.status_label.setWordWrap(True)  # a long status wraps instead of widening the page
        self.status_label.setMinimumWidth(120)
        self.context_label = CaptionLabel(header)
        self.context_bar = ProgressBar(header)
        self.context_bar.setRange(0, 1000)
        self.context_bar.setValue(0)
        self.context_bar.setFixedWidth(180)
        row.addWidget(self.status_label, 1)
        row.addWidget(self.context_label)
        row.addWidget(self.context_bar)
        head.addLayout(row)

        actions_box = QWidget(header)
        actions = FlowLayout(actions_box, isTight=True)  # the buttons wrap on a narrow page
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setHorizontalSpacing(8)
        actions.setVerticalSpacing(6)
        self.stop_button = PushButton(header)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.editor_button = PushButton(header)
        self.editor_button.clicked.connect(self._open_in_editor)
        self.output_button = PushButton(header)
        self.output_button.clicked.connect(self._open_output_folder)
        self.open_log_button = PushButton(header)
        self.open_log_button.clicked.connect(self._open_run_log)
        self.live_button = PushButton(header)
        self.live_button.clicked.connect(self._back_to_live)
        self.live_button.hide()
        self.problems_only = CheckBox(header)
        self.problems_only.stateChanged.connect(lambda *_: self._apply_filter())
        for button in (self.stop_button, self.editor_button, self.output_button, self.open_log_button,
                       self.live_button, self.problems_only):
            actions.addWidget(button)
        head.addWidget(actions_box)
        root.addWidget(header)

        # page list | (timeline over the picked row's full text) | preview; on a narrow page the
        # preview goes under the timeline, so the action and the result stay readable (_fit_width)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.page_list = ListWidget(self.splitter)
        self.page_list.setMinimumWidth(140)
        self.page_list.currentItemChanged.connect(self._on_page_selected)
        self.work_splitter = QSplitter(Qt.Orientation.Horizontal, self.splitter)
        timeline_box = QWidget(self.work_splitter)
        timeline_layout = QVBoxLayout(timeline_box)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(6)
        self.timeline = TreeWidget(timeline_box)
        self.timeline.setColumnCount(5)
        self.timeline.setRootIsDecorated(False)
        self.timeline.setUniformRowHeights(True)
        self.timeline.setWordWrap(False)
        self.timeline.currentItemChanged.connect(self._on_row_selected)
        # at the bottom it stays there, also when the timeline gets shorter (the status wraps ...)
        self._timeline_follows = True
        bar = self.timeline.verticalScrollBar()
        bar.valueChanged.connect(lambda *_: setattr(self, "_timeline_follows", self._at_bottom()))
        bar.rangeChanged.connect(lambda _low, high: bar.setValue(high) if self._timeline_follows else None)
        self.row_detail = CaptionLabel(timeline_box)
        """The picked row in full: its action and result, and the agent's reason, wrapped."""
        self.row_detail.setWordWrap(True)
        self.row_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.row_detail.hide()
        timeline_layout.addWidget(self.timeline, 1)
        timeline_layout.addWidget(self.row_detail)
        self.preview = PagePreview(self._t, self.work_splitter)
        self.preview.setMinimumWidth(220)  # not the width of its stage buttons: the splitter may slim it
        self.work_splitter.addWidget(timeline_box)
        self.work_splitter.addWidget(self.preview)
        self.work_splitter.setStretchFactor(0, 1)
        self.splitter.addWidget(self.page_list)
        self.splitter.addWidget(self.work_splitter)
        self.splitter.setStretchFactor(1, 1)
        self._mode = ""
        self._fit_width(1280)
        root.addWidget(self.splitter, 1)

        self.memory_label = CaptionLabel(self)
        root.addWidget(self.memory_label)

    NARROW = 1100
    """Below this page width the page list and the preview get slimmer and the timeline the rest."""
    STACKED = 760
    """Below this the preview goes under the timeline."""

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().resizeEvent(event)
        self._fit_width(self.width())

    def _fit_width(self, width: int) -> None:
        """Wide: page list, timeline and preview side by side. Narrow: the preview under the
        timeline and a slimmer page list, so a row's action and result fit without scrolling."""
        mode = "stacked" if width < self.STACKED else "narrow" if width < self.NARROW else "wide"
        if mode == self._mode:  # the column widths the user dragged stay (they were reset at every resize)
            return
        self._mode = mode
        narrow = mode != "wide"
        self.context_bar.setFixedWidth(110 if narrow else 180)
        clock = self.timeline.fontMetrics().horizontalAdvance("00:00:00") + 18
        page = self.timeline.fontMetrics().horizontalAdvance("0000") + 22
        for column, size in enumerate((clock, page, 130 if narrow else 150, 240 if narrow else 280)):
            self.timeline.setColumnWidth(column, size)
        self.timeline.setColumnHidden(0, narrow)  # the time: in the page's tooltip and the picked row's text
        unlimited = 16777215
        self.page_list.setMaximumWidth(unlimited if mode == "wide" else 150)
        # a manga page is tall: a slim preview beside the timeline still shows it whole
        self.preview.setMaximumWidth(260 if mode == "narrow" else unlimited)
        self.preview.canvas.setMinimumHeight(140 if mode == "stacked" else 260)
        self.work_splitter.setOrientation(Qt.Orientation.Vertical if mode == "stacked" else Qt.Orientation.Horizontal)
        if mode == "wide":
            self.splitter.setSizes([230, max(600, width - 230)])
            self.work_splitter.setSizes([580, 420])
        else:
            rest = max(300, width - 150)
            self.splitter.setSizes([150, rest])
            self.work_splitter.setSizes([3, 2] if mode == "stacked" else [rest - 260, 260])

    def refresh_texts(self) -> None:
        self.title_label.setText(self._t("Agents"))
        self._show_subtitle()
        self.problems_only.setText(self._t("Agents Problems Only"))
        self.open_log_button.setText(self._t("Agents Open Run Log"))
        self.stop_button.setText(self._t("Stopping...") if self._stopping else self._t("Agents Stop"))
        self.editor_button.setText(self._t("Agents Open In Editor"))
        self.editor_button.setToolTip(self._t("Agents Open In Editor Tip"))
        self.output_button.setText(self._t("Agents Open Output Folder"))
        self.live_button.setText(self._t("Agents Back To Live"))
        self.timeline.setHeaderLabels([self._t("Agents Column Time"), self._t("Agents Column Page"),
                                       self._t("Agents Column Action"), self._t("Agents Column Result"),
                                       self._t("Agents Column Reason")])
        if not self._order:
            self.status_label.setText(self._t("Agents Idle"))
            self.memory_label.setText("")
        self._update_context(None)
        self._rebuild_page_list()
        self.preview.refresh_texts()
        self._update_actions()

    def _show_subtitle(self) -> None:
        if self._viewing_log:
            self.subtitle_label.setText(self._t("Agents Viewing Log", file=os.path.basename(self._viewing_log)))
        else:
            self.subtitle_label.setText(self._t("Agents Page Subtitle"))

    # --- actions --------------------------------------------------------------
    def set_task_running(self, running: bool) -> None:
        """The app started or ended a task. The stop button stops an AI Agent run only, and only on
        the live view: a normal translation, or a run log opened meanwhile, has nothing to stop here."""
        self._task_running = bool(running)
        self._stopping = False
        self._agent_task = False  # the AI Agent workflow says so live once it starts
        if not self._task_running:
            if self._stage:  # ended before a chapter started, without saying why
                self._stage = ""
                if not self._viewing_log:
                    self.status_label.setText(self._t("Agents Stage Ended"))
        self.stop_button.setText(self._t("Agents Stop"))
        self._update_stop()

    def _can_stop(self) -> bool:
        return self._task_running and self._agent_task and not self._viewing_log and not self._stopping

    def _update_stop(self) -> None:
        self.stop_button.setEnabled(self._can_stop())
        chapters = self._t("Agent Result Separator").join(c for c in self._chapters.values() if c)
        self.stop_button.setToolTip(self._t("Agents Stop Tip", chapters=chapters or "-") if self._agent_task else "")

    def _on_stop_clicked(self) -> None:
        if not self._can_stop():  # an old run log, or not an AI Agent task
            self._update_stop()
            return
        self._stopping = True
        self.stop_button.setEnabled(False)
        self.stop_button.setText(self._t("Stopping..."))
        self.stop_requested.emit()

    def _current_page(self) -> str:
        """The page the user picked, else the one the preview shows."""
        if self._page_filter != ALL_PAGES:
            return self._page_filter
        return self.preview.shown or ""

    def _open_in_editor(self) -> None:
        image = self.preview.pages.get(self._current_page(), {}).get("image")
        if image:
            self.open_in_editor_requested.emit(image)

    def _output_folder(self) -> str:
        output = self.preview.pages.get(self._current_page(), {}).get("output") or self._last_output
        return os.path.dirname(output) if output else ""

    def _open_output_folder(self) -> None:
        folder = self._output_folder()
        if folder and os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(folder)))

    def _update_actions(self) -> None:
        self.editor_button.setEnabled(bool(self.preview.pages.get(self._current_page(), {}).get("image")))
        self.output_button.setEnabled(bool(self._output_folder()))
        self.live_button.setVisible(bool(self._viewing_log and (self._live_events or self._agent_task)))
        self._update_stop()

    def _log_folder(self) -> str:
        """Where the run log dialog opens: the last log's folder, else the runs of the chapter shown."""
        if self._last_log_dir and os.path.isdir(self._last_log_dir):
            return self._last_log_dir
        images = [info.get("image") for info in self.preview.pages.values() if info.get("image")]
        if self.preview.shown and self.preview.pages.get(self.preview.shown, {}).get("image"):
            images.insert(0, self.preview.pages[self.preview.shown]["image"])
        for image in images:
            runs = os.path.join(os.path.dirname(os.path.abspath(image)), "manga_translator_work", "agent", "runs")
            if os.path.isdir(runs):
                return runs
        return ""

    # --- events -------------------------------------------------------------
    def _on_agent_event(self, event: dict) -> None:  # worker thread
        self.event_received.emit(dict(event))

    def _handle_live(self, event: dict) -> None:
        """A live event: kept for "back to the current run", shown unless an old run log is open."""
        if event.get("type") == "workflow_stage":
            self._on_stage(event)
            self._update_actions()
            if event.get("stage") == "start":
                self.run_started.emit()
            return
        if event.get("type") == "run_start":
            self._agent_task = True
            self._stage = ""
        new_run = event.get("type") == "run_start" and (not event.get("batch") or event.get("batch") != self._live_batch)
        if new_run:
            self._live_events = []
            self._live_batch = event.get("batch")
            if self._viewing_log:  # a new run takes the page back from the old log
                self._viewing_log = ""
                self._batch = None
                self._show_subtitle()
        self._live_events.append(event)
        if not self._viewing_log:
            self.handle_event(event)
        self._update_actions()

    def _on_stage(self, event: dict) -> None:
        """What a run does before its first chapter has a run log: starting, checking the API, the
        preprocess, the backups of a check (``revise``) or of a redo; how it ended when it ended there (stopped,
        failed); and pages it could not put back at its end (``restore_failed``). The stop button
        stays on while the run is in one of the first stages."""
        stage = str(event.get("stage") or "")
        self._agent_task = stage in ("start", "api_check", "preprocess", "revise", "redo")
        if stage == "start":  # a new run: the last run's summary goes
            self._live_events = []
            self._live_batch = None
            if self._viewing_log:  # like a new run_start, it takes the page back from an old log
                self._viewing_log = ""
                self._show_subtitle()
            self._batch = None
            self._reset()
        names = event.get("pages") if stage == "restore_failed" and isinstance(event.get("pages"), list) else []
        texts = {"start": ("Agents Stage Start", {"pages": event.get("pages", 0)}),
                 "api_check": ("Agents Stage API Check", {"model": event.get("model") or "-"}),
                 "preprocess": ("Agents Stage Preprocess", {"pages": event.get("pages", 0)}),
                 "revise": ("Agents Stage Revise", {"pages": event.get("pages", 0)}),
                 "redo": ("Agents Stage Redo", {"pages": event.get("pages", 0)}),
                 "stopped": ("Agents Stage Stopped", {}),
                 "failed": ("Agents Stage Failed", {"error": event.get("error") or ""}),
                 "restore_failed": ("Agents Stage Restore Failed", {
                     "count": len(names), "pages": self._t("Agent Result Separator").join(map(str, names))})}
        if stage not in texts:
            return
        key, fields = texts[stage]
        text = self._t(key, **fields)
        self._stage = "" if stage in ("stopped", "failed", "restore_failed") else text
        if not self._viewing_log:
            self.status_label.setText(text)

    def _back_to_live(self) -> None:
        self._viewing_log = ""
        self._batch = None
        self._show_subtitle()
        if self._stage and not self._live_events:  # no chapter has started yet
            self._reset()
            self.status_label.setText(self._stage)
        self.replay(list(self._live_events))
        self._update_actions()

    def handle_event(self, event: dict) -> None:
        self._handle(event)
        self._update_actions()

    def _handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "run_start":
            self._start_chapter(event)
            return
        page = self._key(event)
        run = str(event.get("run") or "")
        if kind in ("page_start", "page_done", "inpaint_done", "render_done") or event.get("boxes") is not None:
            # a render that failed or was skipped made no output: the preview kept a missing or an old
            # picture for it (review 2026-11-28 B19)
            saved = kind != "render_done" or str(event.get("result") or "saved") == "saved"
            self.preview.update_page(page, image=event.get("image"), boxes=event.get("boxes"),
                                     inpainted=event.get("inpainted"), output=event.get("output") if saved else None)
        if kind == "page_start":
            self.preview.follow(page)
            self._set_page(page, state="working")
            self.status_label.setText(self._t("Agents Page Progress", current=int(event.get("index", 0)) + 1,
                                              total=event.get("total", len(self._order)), page=self._label(page)))
        elif kind == "tool_call":
            tool = event.get("tool", "")
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            ids = args.get("ids") if isinstance(args.get("ids"), list) and args.get("ids") else [None]
            row = self._add_row(event, self._t(TOOL_KEYS.get(tool, tool)) if tool in TOOL_KEYS else tool,
                                "", event.get("reason", ""), region=_own_region(event, args.get("id", ids[0])))
            row.setToolTip(2, json.dumps(event.get("args", {}), ensure_ascii=False)[:2000])
            # by agent too: the inpaint review and the proofreader flag the same page at the same time
            self._pending_rows.setdefault(f"{page}|{_agent(event)}|{tool}", []).append(row)
        elif kind == "tool_result":
            queue = self._pending_rows.get(f"{page}|{_agent(event)}|{event.get('tool', '')}")
            row = queue.pop(0) if queue else None
            if row is not None:
                text, tip = result_text(self._t, event.get("tool", ""), event.get("summary", ""),
                                        event.get("ok", True), TOOL_KEYS, agent=_agent(event))
                if not (event.get("ok", True) and row.data(0, _FLAG_TEXT)):
                    row.setText(3, text)
                row.setToolTip(3, tip)
                if not event.get("ok", True):
                    row.setData(0, Qt.ItemDataRole.UserRole + 1, True)
                    self._paint(row, WARN)
                self._apply_filter_to(row)
        elif kind == "flag":
            region = event.get("region")
            detail = flag_text(self._t, region, event.get("flag", ""), _agent(event))
            pending = self._pending_rows.get(f"{page}|{_agent(event)}|flag_for_review")
            if pending:  # the flag_for_review call already has a row: highlight it instead of adding a duplicate
                row = pending[0]
                row.setText(3, detail)
                row.setData(0, Qt.ItemDataRole.UserRole + 1, True)
                row.setData(0, _FLAG_TEXT, True)
                self._apply_filter_to(row)
            else:
                row = self._add_row(event, self._t("Agents Flag"), detail, event.get("reason", ""), problem=True,
                                    region=_own_region(event, region))
            self._paint(row, WARN)
            if page:
                if page not in self._pages:
                    self._set_page(page, state="working")
                info = self._pages[page]
                info["flags"] = info.get("flags", 0) + 1
                if _agent(event) == "proofread":  # saved with the page when the proofreading ends
                    info["proof_unsaved"] = info.get("proof_unsaved", 0) + 1
                self._update_item(page)
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
        elif kind == "render_check":  # the sizes the render draws, before mark_page_done
            ids = sorted({int(i["region"]) for i in event.get("issues") or [] if isinstance(i, dict) and i.get("region")})
            if event.get("ok") is False:
                row = self._add_row(event, self._t("Agents Render Check"), str(event.get("error", "")), "", problem=True)
                self._paint(row, ERROR)
            elif ids:
                detail = self._t("Agents Render Check Detail", count=len(ids)) + " · " + ", ".join(f"#{i}" for i in ids)
                row = self._add_row(event, self._t("Agents Render Check"), detail, "", problem=True)
                self._paint(row, WARN)
            else:
                self._add_row(event, self._t("Agents Render Check"), self._t("Agents Render Check OK"), "")
        elif kind == "inpaint_start":
            self._set_page(page, state="inpainting")
            if not (self._translating or self._proofreading):  # the status line follows the main work
                self.status_label.setText(self._t("Agents Inpaint Progress", page=self._label(page)))
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
            if event.get("skipped") == "edited":  # the user saved the page in the editor meanwhile
                detail = self._t("Agents Inpaint Skipped Edited")
            row = self._add_row(event, self._t("Agents Inpaint"), detail, "", problem=flagged > 0)
            if flagged:
                self._paint(row, WARN)
            self._set_page(page, state=event.get("status") or "done", flags=event.get("flags"),
                           confidence=event.get("confidence"))
            if not self._translating and not self._rendering:
                self.preview.follow(page)
            self._show_finished_if_idle()
        elif kind == "inpaint_changes":  # a check: the inpainted image brought up to the page's new mask
            status = str(event.get("status") or "")
            if status == "working":
                self.status_label.setText(self._t("Agents Inpaint Changes Progress", page=self._label(page)))
            elif status == "patched":
                row = self._add_row(event, self._t("Agents Inpaint Changes"),
                                    self._t("Agents Inpaint Changes Patched", added=event.get("added", 0),
                                            restored=event.get("restored", 0)), "")
                self._paint(row, MUTED)
            elif status in ("error", "no_mask"):
                failed = status == "error"
                row = self._add_row(event, self._t("Agents Inpaint Changes"),
                                    self._t("Agents Inpaint Changes Failed" if failed else "Agents Inpaint Changes Unknown"),
                                    str(event.get("error") or "") if failed else "", problem=True)
                self._paint(row, ERROR if failed else WARN)
                self._set_page(page, state="inpaint_failed" if failed else "inpaint_unknown")
        elif kind == "proofread_start":
            self._proofreading = True
            self.status_label.setText(self._t("Agents Proofread Progress"))
            self._add_row(event, self._t("Agents Proofread"),
                          self._t("Agents Proofread Start", pages=event.get("pages", 0),
                                  regions=event.get("regions", 0)), "")
        elif kind == "proofread_done":
            self._requests += _requests(event)
            self._proofreading = False
            self._proof_saved()
            flagged = int(event.get("flagged") or 0)
            unfinished = event.get("finished") is False and bool(event.get("redo"))  # stopped at its step limit
            note = str(event.get("note") or "")
            if unfinished:
                note = self._t("Agents Proofread Unfinished") + (f" {note}" if note else "")
            row = self._add_row(event, self._t("Agents Proofread"),
                                self._t("Agents Proofread Detail", revised=event.get("revised", 0), flagged=flagged),
                                note, problem=flagged > 0 or unfinished)
            if flagged or unfinished:
                self._paint(row, WARN)
            self._show_finished_if_idle()
        elif kind == "fonts":
            unfixed = int(event.get("unfixed") or 0)
            reasons = []
            for change in event.get("changes") or []:
                if change.get("reason") == "fallback":  # the box keeps its font, a few characters come from another
                    reasons.append(f"#{change.get('region')} " + self._t(
                        "Agents Fonts Fallback", chars=change.get("missing", ""),
                        font=str(change.get("fallback") or "").replace("::", " ")))
                    continue
                why = self._t("Agents Fonts Missing", chars=change["missing"]) if change.get("missing") else \
                    self._t("Agents Fonts Layout") if change.get("reason") == "layout" else ""
                reasons.append(f"#{change.get('region')} → {change.get('to')}" + (f" ({why})" if why else ""))
            counts = {"changed": event.get("changed", 0), "unfixed": unfixed}
            detail = self._t("Agents Fonts Detail Borrowed", borrowed=event["borrowed"], **counts) \
                if event.get("borrowed") else self._t("Agents Fonts Detail", **counts)
            row = self._add_row(event, self._t("Agents Fonts"), detail, "; ".join(reasons), problem=unfixed > 0)
            if unfixed:
                self._paint(row, WARN)
        elif kind == "font_styles":
            styles = event.get("styles") or {}
            self._add_row(event, self._t("Agents Font Styles"),
                          self._t("Agents Font Styles Detail", count=len(styles), file=event.get("file", "")),
                          "; ".join(f"{name} → {str(font).replace('::', ' ')}" for name, font in styles.items()))
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
            pages = int(event.get("pages") or 0)
            skipped = event.get("skipped") or {}
            self._unrendered += len(skipped)
            text = self._t("Agents Render Start", pages=pages)
            reason = ""
            if skipped:  # pages without an agent translation are not typeset
                for stem, status in skipped.items():
                    self._set_page(self._key(event, stem),
                                   state=status if status in PAGE_STATE_KEYS else "not_reached")
                separator = self._t("Agent Result Separator")

                def names(items):
                    return separator.join(f"{stem}（{self._t(PAGE_STATE_KEYS.get(status, 'Agent Page Not Reached'))}）"
                                          for stem, status in items)

                held = [(s, v) for s, v in skipped.items() if v in HELD_STATES + REDO_STATES]
                other = [(s, v) for s, v in skipped.items() if v not in HELD_STATES + REDO_STATES]
                reason = "; ".join(([self._t("Agents Render Skipped", pages=names(other))] if other else []) +
                                   ([self._t("Agents Render Held", pages=names(held))] if held else []))
            row = self._add_row(event, self._t("Agents Render"), text, reason, problem=bool(skipped))
            if skipped:
                self._paint(row, WARN)
            if pages:
                self._rendering[run] = pages
                self.status_label.setText(text)
            elif self._finished is not None:
                self.status_label.setText(self._finished_status())
        elif kind == "render_done":
            result = str(event.get("result") or "saved")  # logs from before the result: saved
            output = event.get("output") or ""
            if result == "saved":
                self.preview.follow(page)
                self._last_output = output or self._last_output
                row = self._add_row(event, self._t("Agents Render"),
                                    self._t("Agents Render Detail", file=os.path.basename(output)), "")
                row.setToolTip(3, output)
                self._paint(row, MUTED)
            else:
                if result == "skipped":
                    self._render_skipped += 1
                    reason = str(event.get("reason") or "")
                    text = self._t("Agents Render Skipped Existing", file=os.path.basename(output)) \
                        if reason.startswith("existing_output") else \
                        self._t("Agents Render Skipped Other", reason=event.get("message") or reason or "-")
                    color = WARN
                elif result == "failed":
                    self._render_failed += 1
                    text, color = self._t("Agents Render Failed", error=event.get("error") or "-"), ERROR
                else:
                    text, color = self._t("Agents Render Stopped"), MUTED
                row = self._add_row(event, self._t("Agents Render"), text, event.get("message") or "",
                                    problem=result in ("skipped", "failed"))
                if output:
                    row.setToolTip(3, output)
                self._paint(row, color)
            left = self._rendering.get(run, 0) - 1
            if left > 0:
                self._rendering[run] = left
                self.status_label.setText(self._t("Agents Render Progress", page=self._label(page)))
            else:
                self._rendering.pop(run, None)
                if self._finished is not None and not self._rendering:
                    self.status_label.setText(self._finished_status())
        elif kind == "error":
            error = str(event.get("error", ""))
            if event.get("stage") == "preprocess":  # the agent left the page as it was before the run
                error = self._t("Agent Error Preprocess", reason=error.split("preprocess failed: ", 1)[-1])
                if page:
                    self._set_page(page, state="error")
            hint = ""
            if event.get("redo"):  # the step is written down: starting the chapter again offers to redo it
                hint = self._t("Agents Redo Hint Resets", time=event["resets"]) if event.get("resets") else \
                    self._t("Agents Redo Hint")
            row = self._add_row(event, self._t("Agents Error"), error, hint, problem=True)
            self._paint(row, ERROR)
            if event.get("agent") == "proofread":
                self._proofreading = False
                self._proof_saved()
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
            if not event.get("redo"):  # a redo's are the ones its inpaint_done and proofread_done told
                # (counted twice before: 5 requests showed as 10) (review 2026-11-28 B19)
                self._requests += int(totals.get("requests") or 0) + int(totals.get("recall_requests") or 0)
            # ``total``: every page of the run, also the ones a stopped run never reached
            self._finished_runs[run] = {"done": int(event.get("done") or 0),
                                        "pages": int(event.get("total", event.get("pages", 0)) or 0)}
            self._finished = {key: sum(r[key] for r in self._finished_runs.values()) for key in ("done", "pages")}
            for key in self._run_pages(run):
                if self._pages[key].get("state") == "waiting":
                    self._set_page(key, state="not_reached")
            self._update_memory(event.get("memory"))
            if event.get("reason") == "stopped":
                self._mark_stopped(run)
                return
            self._failed = self._failed or event.get("reason") == "failed"
            reviewing = [p for p in self._order if self._pages.get(p, {}).get("state") == "inpainting"]
            if reviewing:  # the inpaint review of the last page(s) is still running
                self.status_label.setText(self._t("Agents Inpaint Progress", page=self._label(reviewing[0])))
            else:
                self.status_label.setText(self._finished_status())
        elif kind == "stopped":
            self._mark_stopped(run)
        elif kind == "work_memory":
            row = self._add_row(event, self._t("Agents Work Memory"), self._t(
                "Agents Work Memory Saved", added=event.get("glossary_added", 0),
                updated=event.get("glossary_updated", 0), characters=event.get("characters_added", 0)), "")
            row.setToolTip(4, event.get("folder", ""))
            self._paint(row, MUTED)

    def _finished_status(self) -> str:
        """Pages done by the translator, flags as the page list counts them (every agent's), requests."""
        flagged = sum(int(info.get("flags") or 0) for info in self._pages.values())
        if self._finished is not None:
            done, pages = self._finished["done"], self._finished["pages"]
        else:  # stopped before a translator finished
            done = sum(1 for info in self._pages.values() if info.get("state") == "done")
            pages = len(self._order)
        key = "Agents Stopped Status" if self._stopped else "Agents Finished"
        text = self._t(key, done=done, pages=pages, flagged=flagged, requests=self._requests)
        if self._failed:
            text += self._t("Agents Finished Failed")
        if self._unrendered:
            text += self._t("Agents Finished Unrendered", count=self._unrendered)
        if self._render_skipped:
            text += self._t("Agents Finished Render Skipped", count=self._render_skipped)
        if self._render_failed:
            text += self._t("Agents Finished Render Failed", count=self._render_failed)
        return text

    def _mark_stopped(self, run: str) -> None:
        """The user stopped the run: pages being worked on are stopped, the ones after never started."""
        self._stopped = True
        self._translating = self._proofreading = False
        for key in self._run_pages(run):
            state = self._pages[key].get("state")
            if state in ("working", "inpainting"):
                self._set_page(key, state="stopped")
            elif state == "waiting":
                self._set_page(key, state="not_reached")
        if run:
            self._rendering.pop(run, None)
        else:
            self._rendering.clear()
        self.status_label.setText(self._finished_status())

    def _show_finished_if_idle(self) -> None:
        if self._translating or self._proofreading or self._finished is None:
            return
        if not any(info.get("state") == "inpainting" for info in self._pages.values()):
            self.status_label.setText(self._finished_status())

    def replay(self, events: List[dict]) -> None:
        """Many events at once (an old run log, back to the current run). The timeline does not follow
        or repaint each row: it ends at the bottom when it was there or the events started a new run,
        as row by row (1,000 rows took about 0.75 s, mostly scrolling to each one)."""
        follow = self._at_bottom()
        self._replaying, self._replay_reset = True, False
        self.timeline.setUpdatesEnabled(False)
        try:
            for event in events:
                self._handle(event)
        finally:
            self._replaying = False
            self.timeline.setUpdatesEnabled(True)
        if follow or self._replay_reset:
            self.timeline.scrollToBottom()
        self._update_actions()

    # --- helpers --------------------------------------------------------------
    def _start_chapter(self, event: dict) -> None:
        """``run_start``: a new run clears the page; the next chapter of the same run is added to it."""
        run = str(event.get("run") or "")
        batch = event.get("batch") or None
        if batch is None or batch != self._batch or run in self._chapters:
            self._reset()
        self._batch = batch
        self._prefix[run] = f"{run}|" if self._chapters else ""
        self._chapters[run] = str(event.get("chapter") or "")
        names = list(event.get("pages", []))
        stems = event.get("stems")  # the agent's page names (with the extension when two pages share a stem)
        if not isinstance(stems, list) or len(stems) != len(names):
            stems = [os.path.splitext(name)[0] for name in names]
        for stem in stems:
            stem = str(stem)
            key = self._prefix[run] + stem
            if key not in self._pages:
                self._order.append(key)
            self._pages[key] = {"state": "waiting", "flags": 0, "run": run, "stem": stem}
        self._translating = True
        self._proofreading = False
        self._finished = None
        self._update_context(0)
        self._rebuild_page_list()
        self.status_label.setText(self._t("Agents Running", model=event.get("model") or "-"))
        self._update_memory(event.get("memory"))
        redo = event.get("redo")
        if isinstance(redo, dict):  # a redo run: only the steps an earlier run could not do
            row = self._add_row(event, self._t("Agents Redo"), self._t(
                "Agents Redo Start", review=int(redo.get("inpaint_review") or 0),
                proofread=int(redo.get("proofread") or 0)), "")
            self._paint(row, MUTED)
        seeded = event.get("work_memory")
        if isinstance(seeded, dict):  # the chapter's first run starts with what the earlier chapters learned
            row = self._add_row(event, self._t("Agents Work Memory"), self._t(
                "Agents Work Memory Seeded", glossary=seeded.get("glossary", 0),
                characters=seeded.get("characters", 0), story=seeded.get("story", 0)),
                self._t("Agent Result Separator").join(seeded.get("from") or []))
            self._paint(row, MUTED)

    def _reset(self) -> None:
        self._replay_reset = self._replaying
        self.timeline.clear()
        self.row_detail.hide()
        self._pending_rows.clear()
        self._pages = {}
        self._order = []
        self._items = {}
        self._chapters = {}
        self._prefix = {}
        self._finished_runs = {}
        self._finished = None
        self._requests = 0
        self._unrendered = 0
        self._render_skipped = self._render_failed = 0
        self._stopped = self._failed = False
        self._page_filter = ALL_PAGES
        self._rendering.clear()
        self.preview.reset()
        self.preview.set_following(True)
        self._update_context(0)
        self._rebuild_page_list()

    def _key(self, event: dict, stem: Optional[str] = None) -> str:
        """The page's key: its file name, after its run id from the second chapter of a run on."""
        stem = (event.get("page") or "") if stem is None else stem
        return self._prefix.get(str(event.get("run") or ""), "") + stem if stem else ""

    def _label(self, key: str) -> str:
        """The page's name for the status line and the preview: the chapter too when there are several."""
        info = self._pages.get(key)
        if not info:
            return key
        stem = info.get("stem") or key
        chapter = self._chapters.get(info.get("run", ""), "") if len(self._chapters) > 1 else ""
        return f"{chapter} · {stem}" if chapter else stem

    def _run_pages(self, run: str) -> List[str]:
        """The pages of one chapter (all of them for an event without a run id)."""
        return [key for key in self._order if not run or self._pages[key].get("run", "") == run]

    def _add_row(self, event: dict, action: str, detail: str, reason: str, problem: bool = False,
                 region=None) -> QTreeWidgetItem:
        ts = str(event.get("ts", ""))
        clock = ts[11:19] if len(ts) >= 19 else ts
        key = self._key(event)
        row = QTreeWidgetItem([clock, event.get("page") or "", action, detail, reason or ""])
        row.setToolTip(1, " · ".join(t for t in (clock, self._label(key) if key else "") if t))
        row.setToolTip(3, detail)
        row.setToolTip(4, reason or "")
        row.setData(0, Qt.ItemDataRole.UserRole, key)
        row.setData(0, Qt.ItemDataRole.UserRole + 1, problem)
        row.setData(0, Qt.ItemDataRole.UserRole + 2, _region_id(region))
        if self._replaying:  # replay() scrolls once at the end
            self.timeline.addTopLevelItem(row)
            self._apply_filter_to(row)
            return row
        at_bottom = self._at_bottom()  # the user scrolled up to read: stay there
        self.timeline.addTopLevelItem(row)
        self._apply_filter_to(row)
        if at_bottom and not row.isHidden():
            self.timeline.scrollToItem(row)
        return row

    def _at_bottom(self) -> bool:
        """The timeline shows its last row: within a row's height (it scrolls by pixels)."""
        bar = self.timeline.verticalScrollBar()
        slack = 1
        if self.timeline.verticalScrollMode() == self.timeline.ScrollMode.ScrollPerPixel:
            slack = max(2, self.timeline.sizeHintForRow(0)) if self.timeline.topLevelItemCount() else 2
        return bar.value() >= bar.maximum() - slack

    @staticmethod
    def _paint(row: QTreeWidgetItem, status: str) -> None:
        """Color the row by its status (``ui.status_colors``: WARN, ERROR, MUTED) in the theme's color."""
        paint_status(row, status)

    def _apply_filter_to(self, row: QTreeWidgetItem) -> None:
        page_ok = self._page_filter == ALL_PAGES or row.data(0, Qt.ItemDataRole.UserRole) == self._page_filter
        problem_ok = not self.problems_only.isChecked() or bool(row.data(0, Qt.ItemDataRole.UserRole + 1))
        row.setHidden(not (page_ok and problem_ok))

    def _apply_filter(self) -> None:
        for index in range(self.timeline.topLevelItemCount()):
            self._apply_filter_to(self.timeline.topLevelItem(index))

    def _proof_saved(self) -> None:
        """The proofreading ended: its flags are in the pages' sidecars, so in the counts they send."""
        for info in self._pages.values():
            info.pop("proof_unsaved", None)

    def _set_page(self, page: str, state: str, flags=None, confidence=None) -> None:
        if not page:
            return
        added = page not in self._pages
        if added:
            self._order.append(page)
            run = next((r for r, prefix in self._prefix.items() if prefix and page.startswith(prefix)),
                       next((r for r, prefix in self._prefix.items() if not prefix), ""))
            self._pages[page] = {"state": state, "flags": 0, "run": run,
                                 "stem": page[len(self._prefix.get(run, "")):]}
        info = self._pages[page]
        info["state"] = state
        if flags is not None:
            # the count of the page's sidecar: a proofreading that runs beside the inpaint review saves its
            # flags when it ends, and the count went down meanwhile (review 2026-11-28 B19)
            info["flags"] = int(flags) + int(info.get("proof_unsaved") or 0)
        if confidence is not None:
            info["confidence"] = float(confidence)
        if added:
            self._rebuild_page_list()
        else:
            self._update_item(page)

    def _page_text(self, key: str) -> str:
        info = self._pages.get(key, {})
        state = self._t(PAGE_STATE_KEYS.get(info.get("state", "waiting"), "Agent Page Waiting"))
        text = f"{info.get('stem') or key} · {state}"
        if info.get("confidence") is not None:
            text += " · " + self._t("Agents Page Confidence", percent=round(info["confidence"] * 100))
        if info.get("flags"):
            text += " · " + self._t("Agents Flag Count", count=info["flags"])
        return text

    def _update_item(self, key: str) -> None:
        """A page's row changes in place: the list keeps its scroll position and selection."""
        item = self._items.get(key)
        if item is None:
            self._rebuild_page_list()
            return
        item.setText(self._page_text(key))
        state = self._pages.get(key, {}).get("state")
        if state in ("error", "incomplete", "stopped", "redo_failed", *HELD_STATES):
            paint_status(item, ERROR if state in ("error", "inpaint_failed") else WARN)
        else:
            paint_status(item, None)

    def _groups(self) -> List[tuple]:
        """(chapter name or None, page keys): one group per chapter when the run has several."""
        if len(self._chapters) <= 1:
            return [(None, list(self._order))]
        groups = [(name or run, [k for k in self._order if self._pages[k].get("run") == run])
                  for run, name in self._chapters.items()]
        rest = [k for k in self._order if self._pages[k].get("run") not in self._chapters]
        return groups + ([(None, rest)] if rest else [])

    def _rebuild_page_list(self) -> None:
        bar = self.page_list.verticalScrollBar()
        scroll = bar.value()
        self.page_list.blockSignals(True)
        try:
            self.page_list.clear()
            self._items = {}
            everything = QListWidgetItem(self._t("Agents All Pages"))
            everything.setData(Qt.ItemDataRole.UserRole, ALL_PAGES)
            self.page_list.addItem(everything)
            for chapter, keys in self._groups():
                if chapter is not None:  # a heading, not a page: it cannot be selected
                    heading = QListWidgetItem(chapter)
                    heading.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    font = heading.font()
                    font.setBold(True)
                    heading.setFont(font)
                    paint_status(heading, MUTED)
                    self.page_list.addItem(heading)
                for key in keys:
                    item = QListWidgetItem()
                    item.setData(Qt.ItemDataRole.UserRole, key)
                    self._items[key] = item
                    self._update_item(key)
                    self.page_list.addItem(item)
                    if key == self._page_filter:
                        self.page_list.setCurrentItem(item)
            if self._page_filter == ALL_PAGES:
                self.page_list.setCurrentItem(everything)
        finally:
            self.page_list.blockSignals(False)
        self.page_list.doItemsLayout()
        bar.setValue(scroll)

    def _on_page_selected(self, current, _previous) -> None:
        if current is None:
            return
        self._page_filter = current.data(Qt.ItemDataRole.UserRole) or ALL_PAGES
        self._apply_filter()
        if self._page_filter == ALL_PAGES:
            self.preview.set_following(True)
        else:
            self.preview.show_page(self._page_filter)
        self._update_actions()

    def _on_row_selected(self, current, _previous) -> None:
        """A timeline row: its full text under the timeline; its page, and its region when it has one."""
        if current is None:
            self.row_detail.hide()
            return
        head = " · ".join(t for t in (current.text(0), current.text(1), current.text(2), current.text(3)) if t)
        reason = current.text(4)
        self.row_detail.setText(head + (f"\n{reason}" if reason else ""))
        self.row_detail.show()
        page = current.data(0, Qt.ItemDataRole.UserRole) or ""
        if page and page in self.preview.pages:
            self.preview.show_page(page, highlight=current.data(0, Qt.ItemDataRole.UserRole + 2))
            self._update_actions()

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
        path, _ = QFileDialog.getOpenFileName(self, self._t("Agents Open Run Log"), self._log_folder(),
                                              "Agent run log (*.jsonl)")
        if path:
            self.open_run_log(path)

    def open_run_log(self, path: str) -> None:
        """Show an old run; live events of a run going on meanwhile are kept for "back to the current run"."""
        try:
            events = agent_events.read_events(path)
        except OSError as exc:
            self.status_label.setText(str(exc))
            return
        self._last_log_dir = os.path.dirname(os.path.abspath(path))
        self._viewing_log = path
        self._show_subtitle()
        self._batch = None  # a log on its own, not the next chapter of the run shown now
        self.replay(events)
