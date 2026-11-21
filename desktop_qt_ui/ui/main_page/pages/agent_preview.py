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
from qfluentwidgets import BodyLabel, CaptionLabel, CheckBox, FluentIcon, SegmentedWidget, TransparentToolButton

ORIGINAL, INPAINTED, RESULT, COMPARE = "original", "inpainted", "result", "compare"
STAGES = (ORIGINAL, INPAINTED, RESULT, COMPARE)
STAGE_KEYS = {
    ORIGINAL: "Agents Preview Original",
    INPAINTED: "Agents Preview Inpainted",
    RESULT: "Agents Preview Result",
    COMPARE: "Agents Preview Compare",
}
BOX_COLORS = {
    "pending": QColor(220, 50, 50),  # red, like the boxes the agent sees
    "done": QColor(40, 170, 90),
    "skipped": QColor(130, 130, 130),
    "flagged": QColor(214, 140, 0),
}
BOX_MARKS = {"pending": "○", "done": "✓", "flagged": "!", "skipped": "–"}
"""Before the box number, so the state reads without telling the colours apart."""
BOX_LINES = {
    "pending": (Qt.PenStyle.DotLine, 2),
    "done": (Qt.PenStyle.SolidLine, 2),
    "flagged": (Qt.PenStyle.SolidLine, 4),
    "skipped": (Qt.PenStyle.DashLine, 2),
}
"""Line style and width of each state: different in grayscale too."""
HIGHLIGHT = QColor(30, 120, 230)
MAX_SIDE = 2000
"""Images are decoded at most this large; double-clicking opens the full file."""
MAX_ZOOM = 8.0
"""Zoom of the preview over the fitted page."""
ZOOM_STEP = 1.25
"""Zoom factor of one wheel notch."""
PAN_STEP = 48
"""Pixels an arrow key moves a zoomed image."""


def box_label(state: str, rid) -> str:
    return f"{BOX_MARKS.get(state, BOX_MARKS['pending'])}{rid}"


def box_pen(state: str) -> QPen:
    style, width = BOX_LINES.get(state, BOX_LINES["pending"])
    pen = QPen(BOX_COLORS.get(state, BOX_COLORS["pending"]), width)
    pen.setStyle(style)
    return pen


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
    the original left of a divider the user drags.

    The wheel zooms in at the cursor (up to ``MAX_ZOOM`` times the fitted page); a zoomed image
    moves with a drag (with the right or middle button while comparing, where the left button moves
    the divider); a right click without moving fits the page again. The zoom is kept when the stage
    changes, so the same spot can be compared, and reset for another page. With the keyboard focus
    (Tab, or a click): + and - zoom at the middle, 0 fits the page, the arrow keys move."""

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
        self.zoom = 1.0
        self.center = QPointF(0.5, 0.5)
        """The image point (0-1 of its width and height) at the middle of the widget when zoomed."""
        self._pan_from: Optional[QPointF] = None
        self._pan_button = Qt.MouseButton.NoButton
        self._pan_moved = False
        self.setMinimumSize(220, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

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
        """Where the whole image is drawn; larger than the widget when zoomed in."""
        if self.main is None or self.main.width() <= 0 or self.main.height() <= 0:
            return QRectF()
        scale = min(self.width() / self.main.width(), self.height() / self.main.height()) * self.zoom
        w, h = self.main.width() * scale, self.main.height() * scale
        x, y = self.width() / 2 - self.center.x() * w, self.height() / 2 - self.center.y() * h
        # centered while it fits, otherwise no empty margin on either side
        x = (self.width() - w) / 2 if w <= self.width() else min(0.0, max(self.width() - w, x))
        y = (self.height() - h) / 2 if h <= self.height() else min(0.0, max(self.height() - h, y))
        return QRectF(x, y, w, h)

    def view_rect(self) -> QRectF:
        """The part of the image that is on screen."""
        return self.image_rect().intersected(QRectF(self.rect()))

    def _keep_center(self, rect: QRectF) -> None:
        if rect.width() > 0 and rect.height() > 0:
            self.center = QPointF((self.width() / 2 - rect.left()) / rect.width(),
                                  (self.height() / 2 - rect.top()) / rect.height())

    def reset_zoom(self) -> None:
        self.zoom = 1.0
        self.center = QPointF(0.5, 0.5)
        self.update()

    def zoom_at(self, pos: QPointF, steps: float) -> None:
        """Zoom by ``ZOOM_STEP`` per step, keeping the image point under ``pos`` where it is."""
        rect = self.image_rect()
        zoom = min(MAX_ZOOM, max(1.0, self.zoom * ZOOM_STEP ** steps))
        if rect.isEmpty() or zoom == self.zoom:
            return
        px = min(1.0, max(0.0, (pos.x() - rect.left()) / rect.width()))
        py = min(1.0, max(0.0, (pos.y() - rect.top()) / rect.height()))
        w, h = rect.width() * zoom / self.zoom, rect.height() * zoom / self.zoom
        self.zoom = zoom
        self._keep_center(QRectF(pos.x() - px * w, pos.y() - py * h, w, h))
        self._keep_center(self.image_rect())  # within the image's edges
        self.update()

    def zoom_by(self, steps: float) -> None:
        """Zoom at the middle of the widget (the buttons and the keys)."""
        self.zoom_at(QPointF(self.width() / 2, self.height() / 2), steps)

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt)
        key = event.key()
        moves = {Qt.Key.Key_Left: (PAN_STEP, 0), Qt.Key.Key_Right: (-PAN_STEP, 0),
                 Qt.Key.Key_Up: (0, PAN_STEP), Qt.Key.Key_Down: (0, -PAN_STEP)}
        if self.main is not None:
            if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self.zoom_by(1)
                return
            if key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
                self.zoom_by(-1)
                return
            if key in (Qt.Key.Key_0, Qt.Key.Key_Home):
                self.reset_zoom()
                return
            if key in moves and self.zoom > 1.0:
                self.pan(*moves[key])
                return
        super().keyPressEvent(event)

    def focusInEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802 (Qt)
        super().focusOutEvent(event)
        self.update()

    def pan(self, dx: float, dy: float) -> None:
        rect = self.image_rect()
        if rect.isEmpty() or self.zoom <= 1.0:
            return
        self._keep_center(rect.translated(dx, dy))
        self._keep_center(self.image_rect())
        self.update()

    @staticmethod
    def _draw(painter: QPainter, pixmap: QPixmap, rect: QRectF, target: QRectF) -> None:
        """The part of ``pixmap`` (drawn at ``rect``) that falls in ``target``."""
        target = target.intersected(rect)
        if target.isEmpty():
            return
        sx, sy = pixmap.width() / rect.width(), pixmap.height() / rect.height()
        source = QRectF((target.left() - rect.left()) * sx, (target.top() - rect.top()) * sy,
                        target.width() * sx, target.height() * sy)
        painter.drawPixmap(target, pixmap, source)

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
        view = self.view_rect()
        if self.before is not None:
            x = view.left() + view.width() * self.split  # the divider stays on screen when zoomed
            self._draw(painter, self.before, rect, QRectF(view.left(), view.top(), x - view.left(), view.height()))
            self._draw(painter, self.main, rect, QRectF(x, view.top(), view.right() - x, view.height()))
            painter.setPen(QPen(QColor(255, 255, 255), 3))
            painter.drawLine(QPointF(x, view.top()), QPointF(x, view.bottom()))
            painter.setPen(QPen(HIGHLIGHT, 1))
            painter.drawLine(QPointF(x, view.top()), QPointF(x, view.bottom()))
            painter.setBrush(QBrush(HIGHLIGHT))
            painter.drawEllipse(QPointF(x, view.center().y()), 7, 7)
        else:
            self._draw(painter, self.main, rect, view)
        if self.source.width() > 0 and (self.boxes or self.highlight is not None):
            self._paint_boxes(painter, rect, rect.width() / self.source.width())
        if self.hasFocus():  # where the keys go
            painter.setPen(QPen(HIGHLIGHT, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(self.rect()).adjusted(1, 1, -1, -1))

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
            painter.setPen(box_pen(state))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(area)
            self._paint_label(painter, area, box_label(state, rid), color)
        if self.highlight is not None:
            for box in self.boxes:
                if box[0] == self.highlight:
                    rid, x1, y1, x2, y2 = box[:5]
                    state = box[5] if len(box) > 5 else "pending"
                    area = QRectF(rect.left() + x1 * scale, rect.top() + y1 * scale,
                                  (x2 - x1) * scale, (y2 - y1) * scale).adjusted(-3, -3, 3, 3)
                    fill = QColor(HIGHLIGHT)
                    fill.setAlpha(50)
                    painter.setPen(QPen(HIGHLIGHT, 3))
                    painter.setBrush(QBrush(fill))
                    painter.drawRect(area)
                    self._paint_label(painter, area, box_label(state, rid), HIGHLIGHT)

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
        view = self.view_rect()
        if self.before is None or view.width() <= 0:
            return
        self.split = min(1.0, max(0.0, (x - view.left()) / view.width()))
        self.update()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120
        if steps and self.main is not None:
            self.zoom_at(event.position(), steps)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:
        button = event.button()
        if button == Qt.MouseButton.LeftButton and self.before is not None:
            self._drag(event.position().x())
        elif button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._pan_from, self._pan_button, self._pan_moved = event.position(), button, False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_from is not None and event.buttons() & self._pan_button:
            delta = event.position() - self._pan_from
            if self._pan_moved or abs(delta.x()) + abs(delta.y()) > 3:
                self._pan_moved = True
                self._pan_from = event.position()
                self.pan(delta.x(), delta.y())
        elif event.buttons() & Qt.MouseButton.LeftButton:
            self._drag(event.position().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and not self._pan_moved:
            self.reset_zoom()  # a right click fits the page again
        if event.button() == self._pan_button:
            self._pan_from, self._pan_button = None, Qt.MouseButton.NoButton
        super().mouseReleaseEvent(event)

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
        self.label_for: Callable[[str], str] = lambda page: page
        """The page's name in the title; the Agents page keys pages by chapter when a run has several."""
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
        # zoom without the wheel: buttons the Tab key reaches (the canvas takes + - 0 too)
        self.zoom_out_button = TransparentToolButton(FluentIcon.ZOOM_OUT, self)
        self.zoom_out_button.clicked.connect(lambda: self.canvas.zoom_by(-1))
        self.zoom_in_button = TransparentToolButton(FluentIcon.ZOOM_IN, self)
        self.zoom_in_button.clicked.connect(lambda: self.canvas.zoom_by(1))
        self.zoom_fit_button = TransparentToolButton(FluentIcon.FIT_PAGE, self)
        self.zoom_fit_button.clicked.connect(self.canvas.reset_zoom)
        for button in (self.zoom_out_button, self.zoom_in_button, self.zoom_fit_button):
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            foot.addWidget(button)
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
        self.canvas.setAccessibleName(self._t("Agents Preview"))
        for button, key in ((self.zoom_out_button, "Agents Preview Zoom Out"),
                            (self.zoom_in_button, "Agents Preview Zoom In"),
                            (self.zoom_fit_button, "Agents Preview Zoom Fit")):
            button.setToolTip(self._t(key))
            button.setAccessibleName(self._t(key))
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
        if page != self.shown:
            self.canvas.reset_zoom()  # the zoom stays while switching stages of one page
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
        self.title_label.setText(f"{self._t('Agents Preview')} · {self.label_for(page)}")
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
        counts = {state: 0 for state in BOX_MARKS}
        for box in shown_boxes or []:
            state = box[5] if len(box) > 5 else "pending"
            counts[state if state in counts else "pending"] += 1
        # what a screen reader says for the boxes (the marks on them say it on screen)
        self.canvas.setAccessibleDescription(self._t("Agents Preview Box Summary", **counts) if shown_boxes else "")
