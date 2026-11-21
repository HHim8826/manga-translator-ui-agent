"""Live preview of one page on the Agents page: the original (with the agent's region
boxes), the inpainted image, the typeset result, and a before/after comparison.

The Agents page feeds it from the run's events: ``page_start``/``tool_result``/``page_done``
carry the image path and the region boxes, ``inpaint_done`` the inpainted image and
``render_done`` the saved result. While "follow" is on it shows the page being worked on and
its newest stage; picking a page, a stage or a timeline row turns it off.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDesktopServices, QFont, QImageReader, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import BodyLabel, CaptionLabel, CheckBox, SegmentedWidget

ORIGINAL, INPAINTED, RESULT, COMPARE = "original", "inpainted", "result", "compare"
STAGES = (ORIGINAL, INPAINTED, RESULT, COMPARE)
STAGE_KEYS = {
    ORIGINAL: "Agents Preview Original",
    INPAINTED: "Agents Preview Inpainted",
    RESULT: "Agents Preview Result",
    COMPARE: "Agents Preview Compare",
}
BOX_COLORS = {
    "pending": QColor(150, 150, 150),
    "done": QColor(40, 170, 90),
    "skipped": QColor(130, 130, 130),
    "flagged": QColor(214, 140, 0),
}
HIGHLIGHT = QColor(30, 120, 230)
MAX_SIDE = 2000
"""Images are decoded at most this large; double-clicking opens the full file."""


class ImageCache:
    """A few decoded pages, keyed by path and modification time (a rewritten file is read again)."""

    def __init__(self, limit: int = 8):
        self.limit = limit
        self._items: "OrderedDict[Tuple[str, float], Tuple[QPixmap, QSize]]" = OrderedDict()

    def get(self, path: Optional[str]) -> Optional[Tuple[QPixmap, QSize]]:
        """(pixmap, size of the file's image) or None when the file cannot be read."""
        if not path:
            return None
        try:
            key = (os.path.normcase(os.path.abspath(path)), os.path.getmtime(path))
        except OSError:
            return None
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        reader = QImageReader(path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid() and max(size.width(), size.height()) > MAX_SIDE:
            reader.setScaledSize(size.scaled(MAX_SIDE, MAX_SIDE, Qt.AspectRatioMode.KeepAspectRatio))
        image = reader.read()
        if image.isNull():
            return None
        if not size.isValid():
            size = image.size()
        self._items[key] = (QPixmap.fromImage(image), size)
        while len(self._items) > self.limit:
            self._items.popitem(last=False)
        return self._items[key]

    def clear(self) -> None:
        self._items.clear()


class PreviewCanvas(QWidget):
    """Paints one image fitted to the widget, the region boxes on top, and in comparison mode
    the original left of a divider the user drags."""

    double_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.main: Optional[QPixmap] = None
        self.before: Optional[QPixmap] = None
        self.source = QSize()
        self.boxes: List[list] = []
        self.highlight: Optional[int] = None
        self.split = 0.5
        self.message = ""
        self.setMinimumSize(220, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_content(self, main: Optional[QPixmap], source: QSize = QSize(), before: Optional[QPixmap] = None,
                    boxes: Optional[List[list]] = None, highlight: Optional[int] = None, message: str = "") -> None:
        self.main = main
        self.before = before
        self.source = source if source.isValid() else (main.size() if main is not None else QSize())
        self.boxes = list(boxes or [])
        self.highlight = highlight
        self.message = message
        if before is not None:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self.unsetCursor()
        self.update()

    def image_rect(self) -> QRectF:
        if self.main is None or self.main.width() <= 0 or self.main.height() <= 0:
            return QRectF()
        scale = min(self.width() / self.main.width(), self.height() / self.main.height())
        w, h = self.main.width() * scale, self.main.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.main is None:
            painter.setPen(QColor(130, 130, 130))
            painter.drawText(self.rect().adjusted(12, 12, -12, -12),
                             int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap), self.message)
            return
        rect = self.image_rect()
        if self.before is not None:
            x = rect.left() + rect.width() * self.split
            painter.save()
            painter.setClipRect(QRectF(rect.left(), rect.top(), x - rect.left(), rect.height()))
            painter.drawPixmap(rect, self.before, QRectF(self.before.rect()))
            painter.restore()
            painter.save()
            painter.setClipRect(QRectF(x, rect.top(), rect.right() - x, rect.height()))
            painter.drawPixmap(rect, self.main, QRectF(self.main.rect()))
            painter.restore()
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.setPen(QPen(HIGHLIGHT, 1))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.setBrush(QBrush(HIGHLIGHT))
            painter.drawEllipse(QPointF(x, rect.center().y()), 7, 7)
        else:
            painter.drawPixmap(rect, self.main, QRectF(self.main.rect()))
        if self.source.width() > 0 and (self.boxes or self.highlight is not None):
            self._paint_boxes(painter, rect, rect.width() / self.source.width())

    def _paint_boxes(self, painter: QPainter, rect: QRectF, scale: float) -> None:
        font = QFont(self.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.85))
        painter.setFont(font)
        for box in self.boxes:
            rid, x1, y1, x2, y2, state = box[:6]
            if rid == self.highlight:
                continue
            color = BOX_COLORS.get(state, BOX_COLORS["pending"])
            area = QRectF(rect.left() + x1 * scale, rect.top() + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale)
            pen = QPen(color, 2)
            if state == "skipped":
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(area)
            self._paint_label(painter, area, f"{rid}", color)
        if self.highlight is not None:
            for box in self.boxes:
                if box[0] == self.highlight:
                    rid, x1, y1, x2, y2 = box[:5]
                    area = QRectF(rect.left() + x1 * scale, rect.top() + y1 * scale,
                                  (x2 - x1) * scale, (y2 - y1) * scale).adjusted(-3, -3, 3, 3)
                    fill = QColor(HIGHLIGHT)
                    fill.setAlpha(50)
                    painter.setPen(QPen(HIGHLIGHT, 3))
                    painter.setBrush(QBrush(fill))
                    painter.drawRect(area)
                    self._paint_label(painter, area, f"{rid}", HIGHLIGHT)

    @staticmethod
    def _paint_label(painter: QPainter, area: QRectF, text: str, color: QColor) -> None:
        metrics = painter.fontMetrics()
        w, h = metrics.horizontalAdvance(text) + 6, metrics.height()
        tag = QRectF(area.left(), area.top() - h if area.top() >= h else area.top(), w, h)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawRect(tag)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(tag, int(Qt.AlignmentFlag.AlignCenter), text)

    def _drag(self, x: float) -> None:
        rect = self.image_rect()
        if self.before is None or rect.width() <= 0:
            return
        self.split = min(1.0, max(0.0, (x - rect.left()) / rect.width()))
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag(event.position().x())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._drag(event.position().x())
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


class PagePreview(QWidget):
    """The preview panel: page name, stage switch, follow and box toggles, the canvas."""

    def __init__(self, translate: Callable[..., str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._t = translate
        self.pages: Dict[str, dict] = {}
        self.shown: Optional[str] = None
        self.stage = ORIGINAL
        self.live: Optional[str] = None
        self.highlight: Optional[int] = None
        self.images = ImageCache()
        self._build()
        self.refresh_texts()

    # --- layout ---------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 0, 0, 0)
        root.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.title_label = BodyLabel(self)
        self.follow_box = CheckBox(self)
        self.follow_box.setChecked(True)
        self.follow_box.stateChanged.connect(self._on_follow_changed)
        self.boxes_box = CheckBox(self)
        self.boxes_box.setChecked(True)
        self.boxes_box.stateChanged.connect(lambda *_: self._render())
        head.addWidget(self.title_label, 1)
        head.addWidget(self.follow_box)
        root.addLayout(head)
        self.stages = SegmentedWidget(self)
        for stage in STAGES:
            # itemClicked passes ``checked`` first; without the placeholder it would replace ``s``
            self.stages.addItem(stage, stage, onClick=lambda checked=False, s=stage: self._on_stage_clicked(s))
        self.stages.setCurrentItem(ORIGINAL)
        root.addWidget(self.stages)
        self.canvas = PreviewCanvas(self)
        self.canvas.double_clicked.connect(self.open_current_file)
        root.addWidget(self.canvas, 1)
        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.path_label = CaptionLabel(self)
        self.path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        foot.addWidget(self.path_label, 1)
        foot.addWidget(self.boxes_box)
        root.addLayout(foot)

    def refresh_texts(self) -> None:
        for stage in STAGES:
            self.stages.setItemText(stage, self._t(STAGE_KEYS[stage]))
        self.follow_box.setText(self._t("Agents Preview Follow"))
        self.boxes_box.setText(self._t("Agents Preview Boxes"))
        self.boxes_box.setToolTip(self._t("Agents Preview Boxes Tip"))
        self.canvas.setToolTip(self._t("Agents Preview Hint"))
        self._render()

    # --- data -------------------------------------------------------------------
    def reset(self) -> None:
        self.pages.clear()
        self.images.clear()
        self.shown = self.live = None
        self.highlight = None
        self.stage = ORIGINAL
        self._render()

    def update_page(self, page: str, **fields) -> None:
        """Merge what an event says about a page (image, boxes, inpainted, output)."""
        if not page:
            return
        info = self.pages.setdefault(page, {})
        before = self.newest_stage(page)
        info.update({k: v for k, v in fields.items() if v is not None})
        if page != self.shown:
            return
        if self.following and self.newest_stage(page) != before:
            self.stage = self.newest_stage(page)
        self._render()

    def follow(self, page: str) -> None:
        """The page the run is working on now; shown while following."""
        if not page:
            return
        self.live = page
        if self.following:
            self._show(page, self.newest_stage(page), None)

    def show_page(self, page: str, highlight: Optional[int] = None) -> None:
        """The user picked a page (or a timeline row of it): stop following and show it."""
        if not page:
            return
        self._set_following(False)
        same = page == self.shown and self.available(page, self.stage)
        if highlight is not None:  # a region: where the agent saw it, unless this page is already open
            stage = self.stage if same and self.stage != COMPARE else ORIGINAL
        else:
            stage = self.stage if same else self.newest_stage(page)
        self._show(page, stage, highlight)

    def set_following(self, on: bool) -> None:
        self._set_following(on)
        if on:
            self._jump_to_live()

    @property
    def following(self) -> bool:
        return self.follow_box.isChecked()

    def newest_stage(self, page: Optional[str]) -> str:
        info = self.pages.get(page or "", {})
        return RESULT if info.get("output") else INPAINTED if info.get("inpainted") else ORIGINAL

    def available(self, page: Optional[str], stage: str) -> bool:
        info = self.pages.get(page or "", {})
        if stage == ORIGINAL:
            return bool(info.get("image"))
        if stage == INPAINTED:
            return bool(info.get("inpainted"))
        if stage == RESULT:
            return bool(info.get("output"))
        return bool(info.get("image")) and bool(info.get("output") or info.get("inpainted"))

    def stage_path(self, page: Optional[str], stage: str) -> Optional[str]:
        info = self.pages.get(page or "", {})
        if stage == COMPARE:
            return info.get("output") or info.get("inpainted")
        key = {ORIGINAL: "image", INPAINTED: "inpainted", RESULT: "output"}.get(stage)
        return info.get(key) if key else None

    # --- interaction ------------------------------------------------------------
    def _set_following(self, on: bool) -> None:
        self.follow_box.blockSignals(True)
        self.follow_box.setChecked(on)
        self.follow_box.blockSignals(False)

    def _on_follow_changed(self, *_args) -> None:
        if self.following:
            self._jump_to_live()

    def _jump_to_live(self) -> None:
        page = self.live or self.shown
        if page:
            self._show(page, self.newest_stage(page), None)

    def _on_stage_clicked(self, stage: str) -> None:
        if stage not in STAGES:
            return
        self._set_following(False)
        self.stage = stage
        self._render()

    def _show(self, page: str, stage: str, highlight: Optional[int]) -> None:
        self.shown = page
        self.stage = stage
        self.highlight = highlight
        self._render()

    def open_current_file(self) -> None:
        path = self.stage_path(self.shown, self.stage)
        if path and os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(path)))

    # --- drawing ----------------------------------------------------------------
    def _render(self) -> None:
        page = self.shown
        for stage in STAGES:
            item = self.stages.widget(stage)
            if item is not None:
                item.setEnabled(self.available(page, stage))
        if page is None:
            self.title_label.setText(self._t("Agents Preview"))
            self.path_label.setText("")
            self.canvas.set_content(None, message=self._t("Agents Preview Empty"))
            return
        stage = self.stage if self.available(page, self.stage) else ORIGINAL
        self.stages.setCurrentItem(stage)
        self.title_label.setText(f"{self._t('Agents Preview')} · {page}")
        info = self.pages.get(page, {})
        path = self.stage_path(page, stage)
        loaded = self.images.get(path)
        self.path_label.setText(os.path.basename(path) if path else "")
        self.path_label.setToolTip(path or "")
        if loaded is None:
            self.canvas.set_content(None, message=self._t("Agents Preview Missing",
                                                         file=os.path.basename(path) if path else page))
            return
        pixmap, size = loaded
        before = None
        if stage == COMPARE:
            original = self.images.get(info.get("image"))
            before = original[0] if original is not None else None
        boxes = info.get("boxes") or []
        shown_boxes = boxes if (stage == ORIGINAL and self.boxes_box.isChecked()) else \
            [b for b in boxes if b and b[0] == self.highlight]
        if stage != ORIGINAL:  # the boxes are in the original's pixels
            original = self.images.get(info.get("image"))
            if original is not None:
                size = original[1]
        self.canvas.set_content(pixmap, size, before=before, boxes=shown_boxes, highlight=self.highlight)
