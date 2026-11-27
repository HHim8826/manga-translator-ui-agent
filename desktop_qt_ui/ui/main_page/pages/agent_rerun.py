"""Asks before an AI Agent run redoes pages an earlier run already did (``manga_translator.agent.resume``).

The preprocess writes every page's JSON again, so a rerun replaces the agent's translations and the
changes made in the editor. When pages of the run already have a translation (an agent result, or
one from another mode or the editor), the user chooses: only the pages that are not done, every
page, checking the finished pages again (``manga_translator.agent.revise``: they keep their
translations and the user's changes), or nothing. When an earlier run could not do a step on some
pages (the inpaint review, the proofreading: most often the API's usage limit), the first choice is to
redo only those steps (``manga_translator.agent.redo``).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

MAX_NAMES = 8
REVISABLE = ("done", "incomplete", "edited")
"""Agent statuses of pages a check can go over: the agent finished them (``edited``: the user saved
the page during a run)."""
REDO = "redo"
"""What ``ask_rerun`` returns as the mode for "redo the steps that were not done"."""
STEP_KEYS = (("inpaint_review", "Agents Inpaint"), ("proofread", "Agents Proofread"))
"""The steps ``redo.py`` keeps, in its order, with the names the question shows."""


def _names(stems: Sequence[str], separator: str) -> str:
    shown = separator.join(stems[:MAX_NAMES])
    return shown + ("…" if len(stems) > MAX_NAMES else "")


def _window(parent) -> Optional[QWidget]:
    """The widget the question shows over. The start button passes the main page's ``MainView``, a
    QObject that is not a widget (a dialog cannot take it as its parent, the run crashed with a
    TypeError): its ``_dialog_parent()`` is the main window. Anything else: no parent."""
    if isinstance(parent, QWidget):
        return parent
    get = getattr(parent, "_dialog_parent", None)
    window = get() if callable(get) else None
    return window if isinstance(window, QWidget) else None


def rerun_question(t: Callable[..., str], states, lock: bool = True) -> Optional[dict]:
    """Texts and page lists of the question; None when no page has a translation yet. A page
    translated outside the AI Agent (another mode, the editor) counts as done: processing only the
    pages not done leaves it alone. ``lock`` is the setting ``revise_lock_user_edits``: what a check
    does with the boxes the user changed."""
    seen = [s for s in states if s.status]
    others = [s for s in states if not s.status and getattr(s, "translated", False)]
    if not seen and not others:
        return None
    done = [s for s in seen if s.status in ("done", "edited")]  # edited: the user saved it during a run
    edited = [s for s in seen if s.edited]
    revisable = [s for s in seen if s.status in REVISABLE]
    redo = [s for s in revisable if getattr(s, "pending", ())]
    separator = t("Agent Result Separator")
    parts = []  # (text, shown): what happened is shown, the rules of the choices are folded
    if seen:
        parts.append((t("Agent Rerun Text", seen=len(seen), pages=len(states), done=len(done)), True))
    if others:
        parts.append((t("Agent Rerun Translated", count=len(others),
                        pages=_names([s.stem for s in others], separator)), True))
    if edited:
        parts.append((t("Agent Rerun Edited", count=len(edited), pages=_names([s.stem for s in edited], separator)),
                      True))
    steps = ""
    if redo:
        steps = separator.join(t(key) for step, key in STEP_KEYS if any(step in s.pending for s in redo))
        parts.append((t("Agent Rerun Redo Note", count=len(redo), steps=steps,
                        pages=_names([s.stem for s in redo], separator)), False))
        from manga_translator.agent.redo import latest_reset, reset_label

        resets = latest_reset(getattr(s, "resets", "") for s in redo)
        if resets:  # still to come (a time that has passed is not shown)
            parts.append((t("Agent Rerun Redo Resets", time=reset_label(resets)), True))
    keep = {id(s) for s in done + others}
    unfinished = [s for s in states if id(s) not in keep]
    if revisable:
        parts.append((t("Agent Rerun Revise Note"), False))
        parts.append((t("Agent Rerun Revise Locked") if lock else t("Agent Rerun Revise Unlocked Note"), False))
        if unfinished:  # two choices that take different pages, maybe as many of them
            parts.append((t("Agent Rerun Pages", revise=_names([s.stem for s in revisable], separator),
                            unfinished=_names([s.stem for s in unfinished], separator)), False))
    parts.append((t("Agent Rerun Stop Note"), False))
    return {"text": "\n\n".join(text for text, _shown in parts),
            "summary": "\n\n".join(text for text, shown in parts if shown),
            "details": "\n\n".join(text for text, shown in parts if not shown),
            "steps": steps, "unfinished": [s.path for s in unfinished],
            "all": [s.path for s in states], "revise": [s.path for s in revisable],
            "redo": [s.path for s in redo],
            "edited": len(edited), "translated": len(others), "lock": lock}


def _exec(dialog) -> None:
    """Shows a question and waits for the answer (the tests answer in its place)."""
    dialog.exec()


SCREEN_MARGIN = 48
"""Room kept free above and below the dialog on the screen."""
MIN_BODY = 120
"""The scrolled part (the choices and the details) is never lower than this."""
WIDTH = 600
"""The dialog's width, unless a choice button needs more."""
SCROLL_ROOM = 16
"""Room beside the choices for their scroll bar."""


def _room(dialog) -> int:
    """The height the dialog may take: its screen's free height (without the taskbar) less a margin."""
    from PyQt6.QtGui import QGuiApplication

    screen = dialog.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return 16777215
    return max(200, screen.availableGeometry().height() - SCREEN_MARGIN)


def _screen_width(dialog) -> int:
    """The width the dialog may take: its screen's free width less a margin."""
    from PyQt6.QtGui import QGuiApplication

    screen = dialog.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return 16777215
    return max(WIDTH // 2, screen.availableGeometry().width() - SCREEN_MARGIN)


def _width(dialog, screen_width: Optional[int] = None) -> int:
    """``WIDTH``, or as wide as the longest choice button (they are one line: in es_ES one was 107 px
    wider than the dialog, and every note and the details were laid out that wide and cut off on the
    right), up to the screen's width; a button still too long there shows its label cut short, all of
    it in its tooltip (review 2026-11-28 U1)."""
    screen_width = _screen_width(dialog) if screen_width is None else screen_width
    margins = dialog.textLayout.contentsMargins()
    extra = margins.left() + margins.right() + SCROLL_ROOM
    buttons = [b for b in getattr(dialog, "choice_buttons", []) if b is not getattr(dialog, "cancelButton", None)]
    for button in buttons:  # measured with the whole label
        full = button.property("fullText")
        if full:
            button.setText(full)
    needed = max((b.sizeHint().width() for b in buttons), default=0) + extra
    width = min(max(WIDTH, needed), screen_width)
    inner = width - extra
    for button in buttons:
        if button.sizeHint().width() <= inner:
            continue
        full = button.text()
        metrics = button.fontMetrics()
        chrome = button.sizeHint().width() - metrics.horizontalAdvance(full)
        button.setProperty("fullText", full)
        button.setToolTip(full)
        button.setText(metrics.elidedText(full, Qt.TextElideMode.ElideRight, max(40, inner - chrome)))
    return width


def _keep_on_screen(dialog) -> None:
    """A shown dialog that grew (the details opened) moved back inside its screen: it went down
    under the taskbar (review 2026-11-28 U1)."""
    from PyQt6.QtGui import QGuiApplication

    if not dialog.isVisible():
        return
    screen = dialog.screen() or QGuiApplication.primaryScreen()
    if screen is None:
        return
    area, frame = screen.availableGeometry(), dialog.frameGeometry()
    x = max(area.left(), min(frame.left(), area.right() + 1 - frame.width()))
    y = max(area.top(), min(frame.top(), area.bottom() + 1 - frame.height()))
    if (x, y) != (frame.left(), frame.top()):
        dialog.move(dialog.x() + x - frame.left(), dialog.y() + y - frame.top())


def _natural(dialog, width: int) -> int:
    layout = dialog.layout()
    if layout is None:
        return 0
    layout.activate()
    height = layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else -1
    return max(height, layout.totalMinimumSize().height())


def _fit(dialog, width: Optional[int] = None, room: Optional[int] = None) -> None:
    """The dialog ``width`` wide (``_width``) and as tall as what it shows (it grows when the details
    open), but no taller than the screen (``room``): then the choices and the details scroll, and
    Cancel stays in sight at the bottom (review 2026-11-26 P2-4)."""
    if width is None:
        width = _width(dialog)
    dialog.setMinimumSize(0, 0)
    dialog.setMaximumSize(16777215, 16777215)
    room = _room(dialog) if room is None else room
    scroll, body = getattr(dialog, "body_scroll", None), getattr(dialog, "body", None)
    if scroll is not None and body is not None:
        margins = dialog.textLayout.contentsMargins()
        inner = width - margins.left() - margins.right()
        layout = body.layout()
        layout.activate()
        wanted = layout.totalHeightForWidth(inner) if layout.hasHeightForWidth() else body.sizeHint().height()
        wanted = max(wanted, layout.totalMinimumSize().height())
        scroll.setFixedHeight(wanted)
        over = _natural(dialog, width) - room
        if over > 0:
            scroll.setFixedHeight(max(MIN_BODY, wanted - over))
    dialog.setFixedSize(width, _natural(dialog, width))
    _keep_on_screen(dialog)


def _dialog(parent, t: Callable[..., str], title: str, summary: str, details: str = "",
            choices: Sequence[Tuple[str, str, bool]] = ()):
    """A question in the app's theme (a qfluentwidgets ``Dialog``, as the app's other dialogs; a
    QMessageBox stayed white in the dark theme): the summary, a row per choice (its button, and
    under it a line on what it does), the long rules folded under a details button, and Cancel.
    ``choices``: (label, what it does, the default). After ``_exec``: ``dialog.chosen`` is the index
    of the choice clicked, None for Cancel (or Esc). ``dialog.choice_buttons`` ends with Cancel."""
    from qfluentwidgets import (CaptionLabel, Dialog, PrimaryPushButton, PushButton, SingleDirectionScrollArea,
                                TransparentPushButton)

    dialog = Dialog(title, summary, parent)
    dialog.setTitleBarVisible(False)
    dialog.contentLabel.setText(summary)  # Dialog breaks long lines by a count of characters
    dialog.contentLabel.setWordWrap(True)
    dialog.setContentCopyable(True)
    dialog.yesButton.hide()
    dialog.cancelButton.setText(t("Cancel"))
    dialog.chosen = None
    dialog.choice_buttons = []
    dialog.default_button = None

    def choose(index: int) -> None:
        dialog.chosen = index
        dialog.accept()

    # the choices and the details scroll when the dialog would be taller than the screen; the summary
    # stays above them and Cancel below
    body = QWidget()
    body.setObjectName("rerunBody")
    body_layout = QVBoxLayout(body)
    body_layout.setContentsMargins(0, 0, 0, 0)
    body_layout.setSpacing(dialog.textLayout.spacing())
    scroll = SingleDirectionScrollArea(dialog, Qt.Orientation.Vertical)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(body)
    scroll.enableTransparentBackground()
    dialog.textLayout.addWidget(scroll)
    dialog.body, dialog.body_scroll = body, scroll

    for index, (label, description, default) in enumerate(choices):
        row = QWidget(body)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)
        button = (PrimaryPushButton if default else PushButton)(label, row)
        button.clicked.connect(lambda _checked=False, i=index: choose(i))
        layout.addWidget(button)
        if description:
            note = CaptionLabel(description, row)
            note.setWordWrap(True)
            layout.addWidget(note)
        body_layout.addWidget(row)
        dialog.choice_buttons.append(button)
        if default and dialog.default_button is None:
            dialog.default_button = button
            button.setDefault(True)
            button.setFocus()
    dialog.details_label = dialog.details_button = None
    if details:
        label = CaptionLabel(details, body)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setVisible(False)
        toggle = TransparentPushButton(t("Agent Rerun Details"), body)

        def flip() -> None:
            label.setVisible(label.isHidden())
            toggle.setText(t("Agent Rerun Details" if label.isHidden() else "Agent Rerun Hide Details"))
            _fit(dialog)

        toggle.clicked.connect(flip)
        body_layout.addWidget(toggle, 0, Qt.AlignmentFlag.AlignLeft)
        body_layout.addWidget(label)
        dialog.details_label, dialog.details_button = label, toggle
    body_layout.addStretch(1)
    dialog.choice_buttons.append(dialog.cancelButton)
    if dialog.default_button is None:
        dialog.default_button = dialog.cancelButton
        dialog.cancelButton.setFocus()
    _fit(dialog)
    return dialog


def ask_rerun(parent, t: Callable[..., str], files: Sequence[str],
              states=None, lock: bool = True) -> Optional[Tuple[List[str], Union[bool, str]]]:
    """(the files to process, the mode: False, True to check them again, or ``REDO`` to redo the steps
    an earlier run could not do) — all of them, only the ones not done, the finished ones checked
    again, or the pages with steps to redo; None when the user cancels. ``lock``: the run's
    ``revise_lock_user_edits``, so the check-again button says what happens to the user's boxes.
    ``parent``: a widget, or the main page's ``MainView`` (see ``_window``)."""
    parent = _window(parent)
    if states is None:
        from manga_translator.agent.resume import previous_states
        states = previous_states(files)
    busy = [s for s in states if getattr(s, "busy", False)]
    if busy:  # another run works on the chapter: a second one would change the same pages
        _exec(_dialog(parent, t, t("Agent Rerun Title"),
                      t("Agent Rerun Busy", pages=_names([s.stem for s in busy], t("Agent Result Separator")))))
        return None
    question = rerun_question(t, states, lock)
    if question is None:
        return list(files), False
    choices = []  # (label, what it does, the default, the answer)
    if question["redo"]:
        choices.append((t("Agent Rerun Redo", count=len(question["redo"])),
                        t("Agent Rerun Redo Desc", steps=question["steps"]), True, (question["redo"], REDO)))
    if question["revise"]:
        choices.append((t("Agent Rerun Revise" if lock else "Agent Rerun Revise Unlocked",
                          count=len(question["revise"])),
                        t("Agent Rerun Revise Desc"), False, (question["revise"], True)))
    if question["unfinished"]:
        choices.append((t("Agent Rerun Unfinished", count=len(question["unfinished"])),
                        t("Agent Rerun Unfinished Desc"), not question["redo"], (question["unfinished"], False)))
    choices.append((t("Agent Rerun All", count=len(question["all"])), t("Agent Rerun All Desc"), False,
                    (question["all"], False)))
    dialog = _dialog(parent, t, t("Agent Rerun Title"), question["summary"], question["details"],
                     [choice[:3] for choice in choices])
    _exec(dialog)
    if dialog.chosen is None:
        return None
    return choices[dialog.chosen][3]
