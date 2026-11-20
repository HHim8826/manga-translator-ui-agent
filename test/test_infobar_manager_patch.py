"""Two toasts on the same parent must not hit KeyError in InfoBarManager.aniGroups.

The safety patch used to pre-create ``infoBars[parent]`` before calling the original
``add()``, which then skipped creating ``aniGroups[parent]``; the second toast on the
same widget (e.g. every editor export) raised ``KeyError: <weakref ... EditorView>``.
"""
import _bootstrap  # noqa: F401

from PyQt6.QtCore import QEvent, QSize, Qt
from PyQt6.QtGui import QResizeEvent, QShowEvent
from PyQt6.QtWidgets import QApplication, QScrollArea, QWidget

app = QApplication.instance() or QApplication([])

from qfluentwidgets import InfoBar, InfoBarIcon, InfoBarPosition  # noqa: E402
from qfluentwidgets.components.widgets.info_bar import InfoBarManager  # noqa: E402

import ui.widgets.toast_notification  # noqa: E402,F401  (installs the patches)


def _bar(parent):
    return InfoBar(
        icon=InfoBarIcon.SUCCESS,
        title="",
        content="x",
        orient=Qt.Orientation.Horizontal,
        isClosable=True,
        duration=-1,
        position=InfoBarPosition.BOTTOM,
        parent=parent,
    )


def test_second_toast_on_same_parent_does_not_raise():
    parent = QWidget()
    parent.resize(400, 300)
    manager = InfoBarManager.make(InfoBarPosition.BOTTOM)
    first, second = _bar(parent), _bar(parent)
    manager.add(first)
    manager.add(second)  # used to raise KeyError
    assert manager.infoBars[parent] == [first, second]
    assert parent in manager.aniGroups


def test_existing_entry_without_animation_group_self_heals():
    parent = QWidget()
    manager = InfoBarManager.make(InfoBarPosition.BOTTOM)
    manager.infoBars[parent] = []  # state the old patch left behind
    manager.add(_bar(parent))
    manager.add(_bar(parent))
    assert parent in manager.aniGroups


def test_scrollbar_resize_filter_tolerates_wrong_event_wrapper():
    from qfluentwidgets.components.widgets.scroll_bar import ScrollBar

    host = QScrollArea()
    host.resize(200, 100)
    bar = ScrollBar(Qt.Orientation.Vertical, host)

    class WrongWrapper(QShowEvent):  # a Resize event that PyQt hands over as QShowEvent
        def type(self):
            return QEvent.Type.Resize

    bar.eventFilter(host, WrongWrapper())  # used to raise AttributeError: no 'size'
    bar.eventFilter(host, QResizeEvent(QSize(200, 100), QSize(10, 10)))
