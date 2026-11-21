"""The editor's "AI Agent" tab: what the AI Agent left on the open page (``manga_translator.agent.review``).

* the review list of every agent; picking one selects its box in the editor (or shows the area);
* the boxes the translator skipped (sound effects, signs...), which are not in the JSON: picking one
  shows where it is, "Restore" adds it back as a box;
* the inpaint review's areas with every attempt: "Use this attempt" puts another one into the
  inpainted image.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import QHBoxLayout, QTreeWidgetItem, QVBoxLayout, QWidget
from qfluentwidgets import CaptionLabel, PushButton, TreeWidget

from manga_translator.agent import review as agent_review

WARN = QColor(214, 140, 0)
MUTED = QColor(130, 130, 130)
ROLE = Qt.ItemDataRole.UserRole
DETAIL = Qt.ItemDataRole.UserRole + 1
CHOSEN_MARK = "✓ "
AGENTS = ("translator", "proofread", "fonts", "inpaint")


class AgentReviewPanel(QWidget):
    region_selected = pyqtSignal(int)
    """An editor region index to select."""
    area_requested = pyqtSignal(list)
    """``[x1, y1, x2, y2]`` in image pixels, to bring into view."""
    restore_requested = pyqtSignal(dict)
    """Region data to add back to the page."""
    attempt_requested = pyqtSignal(str, int)
    """An inpaint area and the attempt to use."""

    def __init__(self, translate: Callable[..., str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._t = translate
        self.review: Optional[agent_review.PageReview] = None
        self.regions: List[dict] = []
        self.image_path: Optional[str] = None
        self._stale = False
        self._build()
        self.refresh_ui_texts()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.summary_label = CaptionLabel(self)
        self.summary_label.setWordWrap(True)
        self.tree = TreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(1)
        self.tree.setIndentation(14)
        self.tree.currentItemChanged.connect(self._on_item_changed)
        # rows stay one short line in the narrow panel; the picked row's whole text and reason show here
        self.detail_label = CaptionLabel(self)
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.detail_label.hide()
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.restore_button = PushButton(self)
        self.restore_button.clicked.connect(self._restore)
        self.attempt_button = PushButton(self)
        self.attempt_button.clicked.connect(self._use_attempt)
        buttons.addWidget(self.restore_button)
        buttons.addWidget(self.attempt_button)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.detail_label)
        layout.addLayout(buttons)

    def refresh_ui_texts(self) -> None:
        self.restore_button.setText(self._t("Agent Review Restore"))
        self.restore_button.setToolTip(self._t("Agent Review Restore Tip"))
        self.attempt_button.setText(self._t("Agent Review Use Attempt"))
        self._fill()

    # --- data -------------------------------------------------------------------
    def set_page(self, image_path: Optional[str], regions: List[dict]) -> None:
        """The editor opened a page (or reloaded it)."""
        self.image_path = image_path
        self.regions = list(regions or [])
        try:
            self.review = agent_review.load(image_path) if image_path else None
        except Exception:  # a broken sidecar must not break the editor
            self.review = None
        self._fill()

    def set_regions(self, regions: List[dict]) -> None:
        """The page's boxes changed: which flags still find their box (redrawn when the tab shows)."""
        self.regions = list(regions or [])
        if self.review is None:
            return
        if not self.isVisible():
            self._stale = True
            return
        self._fill(select=self._key())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._stale:
            self._fill(select=self._key())

    def message(self, text: str) -> None:
        self.summary_label.setText(text)

    # --- list ---------------------------------------------------------------------
    def _summary(self) -> str:
        review = self.review
        if review is None:
            return self._t("Agent Review Empty")
        from ui.main_page.pages.agents_page import PAGE_STATE_KEYS

        # "pending": the sidecar was saved before the agent marked the page done
        state = {**PAGE_STATE_KEYS, "pending": "Agent Page Incomplete"}.get(review.status)
        parts = [self._t(state) if state else ""]
        if review.confidence is not None:
            parts.append(self._t("Agents Page Confidence", percent=round(review.confidence * 100)))
        text = self._t("Agent Review Summary", state=" · ".join(p for p in parts if p))
        return text + (f"\n{review.note}" if review.note else "")

    def _group(self, title: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([title])
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        self.tree.addTopLevelItem(item)
        return item

    @staticmethod
    def _row(parent: QTreeWidgetItem, text: str, key, detail: str = "", color: Optional[QColor] = None):
        item = QTreeWidgetItem([text])
        item.setData(0, ROLE, key)
        full = "\n".join(p for p in (text, detail) if p)
        item.setData(0, DETAIL, full)
        item.setToolTip(0, full)
        if color is not None:
            item.setForeground(0, QBrush(color))
        parent.addChild(item)
        return item

    def _fill(self, select=None) -> None:
        from ui.main_page.pages.agent_results import ISSUES, SKIP_KINDS, _known, method_name

        self._stale = False
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            self.summary_label.setText(self._summary())
            review = self.review
            if review is None:
                return
            chosen = None
            flags = self._group(self._t("Agent Review Flags", count=len(review.flags)))
            for index, flag in enumerate(review.flags):
                item = self._row(flags, self._flag_text(flag), ("flag", index), flag.reason, WARN)
                chosen = item if select == ("flag", index) else chosen
            skipped = self._group(self._t("Agent Review Skipped", count=len(review.skipped)))
            for index, box in enumerate(review.skipped):
                kind = _known(self._t, "Agent Skip", box.category, SKIP_KINDS)
                text = f"#{box.id} {kind}" + (f" · {box.text}" if box.text else "")
                if self._restored(box):
                    text += " · " + self._t("Agent Review Restored")
                item = self._row(skipped, text, ("skipped", index), box.reason, MUTED)
                chosen = item if select == ("skipped", index) else chosen
            units = self._group(self._t("Agent Review Inpaint", count=len(review.units)))
            for unit in review.units:
                text = self._t("Agent Review Unit", id=unit.id, chosen=unit.chosen, count=len(unit.attempts))
                parent = self._row(units, text, ("unit", unit.id), self._t("Agent Review Flagged") if unit.flagged else "",
                                   WARN if unit.flagged else None)
                chosen = parent if select == ("unit", unit.id) else chosen
                for attempt in unit.attempts:
                    verdict = self._t("Agent Result Separator").join(
                        _known(self._t, "Agent Issue", v, ISSUES) for v in attempt.verdict if v != "pass"
                    ) or self._t("Agent Result No Issues")
                    label = self._t("Agent Review Attempt", n=attempt.n, method=method_name(self._t, attempt.method),
                                    verdict=verdict)
                    in_use = attempt.n == unit.chosen
                    child = self._row(parent, (CHOSEN_MARK if in_use else "") + label, ("attempt", unit.id, attempt.n),
                                      self._t("Agent Review Chosen") if in_use else "")
                    chosen = child if select == ("attempt", unit.id, attempt.n) else chosen
            self.tree.expandAll()
            if chosen is not None:
                self.tree.setCurrentItem(chosen)
        finally:
            self.tree.blockSignals(False)
            self._update_buttons()

    def _flag_text(self, flag) -> str:
        """``翻譯 · 第 4 框：原文不確定``, numbered like the editor's boxes (each agent numbers its own way)."""
        from ui.main_page.pages.agent_results import flag_text

        agent = self._t("Agent Review Agent " + (flag.agent if flag.agent in AGENTS else "translator"))
        found = agent_review.flag_region(flag, self.regions)
        if found is not None:
            what = flag_text(self._t, found + 1, flag.type, "editor")  # 第 4 框, like proofreading on the timeline
        elif flag.agent == "inpaint" and flag.number is not None:
            area = self._t("Agent Review Area", id=flag.number)
            what = area if flag.type == "inpaint" else f"{flag_text(self._t, None, flag.type)} · {area}"
        elif flag.number is not None:
            what = flag_text(self._t, None, flag.type) + " · " + self._t("Agent Review Not Found")
        else:
            what = flag_text(self._t, None, flag.type)
        return f"{agent} · {what}"

    # --- interaction ----------------------------------------------------------------
    def _key(self):
        item = self.tree.currentItem()
        return item.data(0, ROLE) if item is not None else None

    def _restored(self, box) -> bool:
        """A box is back on the page (restored, or drawn by hand at its place)."""
        return agent_review.match_region(None, box.bbox, self.regions) is not None

    def _unit(self, uid: str):
        return next((u for u in self.review.units if u.id == uid), None) if self.review else None

    def _update_buttons(self) -> None:
        self._show_detail()
        key = self._key()
        self.restore_button.setEnabled(bool(key) and key[0] == "skipped" and self.review is not None
                                       and not self._restored(self.review.skipped[key[1]]))
        unit = self._unit(key[1]) if key and key[0] == "attempt" else None
        self.attempt_button.setEnabled(unit is not None and key[2] != unit.chosen)

    def _show_detail(self) -> None:
        item = self.tree.currentItem()
        detail = item.data(0, DETAIL) if item is not None else ""
        self.detail_label.setText(detail or "")
        self.detail_label.setVisible(bool(detail))

    def _on_item_changed(self, current, _previous) -> None:
        self._update_buttons()
        key = current.data(0, ROLE) if current is not None else None
        if not key or self.review is None:
            return
        if key[0] == "flag":
            flag = self.review.flags[key[1]]
            index = agent_review.flag_region(flag, self.regions)
            if index is not None:
                self.region_selected.emit(index)
                box = agent_review.region_bbox(self.regions[index])
                if box is not None:
                    self.area_requested.emit(box)
            elif flag.bbox is not None:
                self.area_requested.emit(list(flag.bbox))
        elif key[0] == "skipped":
            self.area_requested.emit(list(self.review.skipped[key[1]].bbox))
        elif key[0] in ("unit", "attempt"):
            unit = self._unit(key[1])
            if unit is not None:
                self.area_requested.emit([float(v) for v in unit.box])

    def _restore(self) -> None:
        key = self._key()
        if self.review is None or not key or key[0] != "skipped" or self._restored(self.review.skipped[key[1]]):
            return
        self.restore_requested.emit(agent_review.restore_data(self.review.skipped[key[1]], self.regions))

    def _use_attempt(self) -> None:
        key = self._key()
        if key and key[0] == "attempt":
            self.attempt_requested.emit(key[1], int(key[2]))
