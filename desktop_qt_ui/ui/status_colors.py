"""Status colors of the AI Agent page (timeline, page list) and the editor's review list.

One fixed color per status was hard to read in one of the themes (review 2026-11-24 U3: the
orange of "to check" was 2.7:1 on the light list, the red of an error 2.9:1 on the dark one). Each
status has a light and a dark theme color now, with a text contrast of at least 4.5:1 on Fluent's
list backgrounds (#FBFBFB, #2B2B2B). A painted item keeps its status, so the colors follow a theme
change (``repaint``)."""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor

WARN, ERROR, MUTED = "warn", "error", "muted"
COLORS = {  # status: (light theme, dark theme)
    WARN: ((157, 93, 0), (214, 140, 0)),
    ERROR: ((196, 43, 28), (255, 120, 110)),
    MUTED: ((110, 110, 110), (160, 160, 160)),
}
STATUS_ROLE = Qt.ItemDataRole.UserRole + 40
"""Where a painted item keeps its status (column 0)."""


def dark_theme() -> bool:
    try:
        from qfluentwidgets import isDarkTheme

        return bool(isDarkTheme())
    except Exception:
        return False


def status_color(status: str, dark: Optional[bool] = None) -> QColor:
    light, night = COLORS[status]
    return QColor(*(night if (dark_theme() if dark is None else dark) else light))


def paint(item, status: Optional[str], columns: Optional[int] = None) -> None:
    """Color a QTreeWidgetItem's columns (all of them unless ``columns``) or a QListWidgetItem by
    ``status``; None gives the item the theme's text color back."""
    tree = hasattr(item, "columnCount")
    count = (columns if columns is not None else item.columnCount()) if tree else 1
    brush = QBrush(status_color(status)) if status else None
    for column in range(count):
        if tree:
            if brush is None:
                item.setData(column, Qt.ItemDataRole.ForegroundRole, None)
            else:
                item.setForeground(column, brush)
        elif brush is None:
            item.setData(Qt.ItemDataRole.ForegroundRole, None)
        else:
            item.setForeground(brush)
    if tree:
        item.setData(0, STATUS_ROLE, status)
        item.setData(0, STATUS_ROLE + 1, count)
    else:
        item.setData(STATUS_ROLE, status)


def repaint(widget) -> None:
    """The items of a QTreeWidget or QListWidget painted by ``paint``, in the current theme's colors."""
    if hasattr(widget, "topLevelItemCount"):
        from PyQt6.QtWidgets import QTreeWidgetItemIterator

        iterator = QTreeWidgetItemIterator(widget)
        while iterator.value() is not None:
            item = iterator.value()
            status = item.data(0, STATUS_ROLE)
            if status:
                paint(item, status, item.data(0, STATUS_ROLE + 1))
            iterator += 1
    elif hasattr(widget, "count") and hasattr(widget, "item"):
        for index in range(widget.count()):
            item = widget.item(index)
            status = item.data(STATUS_ROLE) if item is not None else None
            if status:
                paint(item, status)


def follow_theme(owner, *widgets) -> None:
    """Repaint ``widgets`` when the app's theme changes, while ``owner`` lives."""
    try:
        from qfluentwidgets import qconfig
    except Exception:
        return

    def update(*_args) -> None:
        for widget in widgets:
            try:
                repaint(widget)
            except RuntimeError:  # deleted
                pass

    qconfig.themeChanged.connect(update)
    owner.destroyed.connect(lambda *_: _disconnect(qconfig.themeChanged, update))


def _disconnect(signal, slot) -> None:
    try:
        signal.disconnect(slot)
    except (TypeError, RuntimeError):
        pass
