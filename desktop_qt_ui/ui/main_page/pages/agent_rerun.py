"""Asks before an AI Agent run redoes pages an earlier run already did (``manga_translator.agent.resume``).

The preprocess writes every page's JSON again, so a rerun replaces the agent's translations and the
changes made in the editor. When pages of the run already have a translation (an agent result, or
one from another mode or the editor), the user chooses: only the pages that are not done, every
page, checking the finished pages again (``manga_translator.agent.revise``: they keep their
translations and the user's changes), or nothing.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from PyQt6.QtWidgets import QMessageBox

MAX_NAMES = 8
REVISABLE = ("done", "incomplete", "edited")
"""Agent statuses of pages a check can go over: the agent finished them (``edited``: the user saved
the page during a run)."""


def _names(stems: Sequence[str], separator: str) -> str:
    shown = separator.join(stems[:MAX_NAMES])
    return shown + ("…" if len(stems) > MAX_NAMES else "")


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
    separator = t("Agent Result Separator")
    parts = []
    if seen:
        parts.append(t("Agent Rerun Text", seen=len(seen), pages=len(states), done=len(done)))
    if others:
        parts.append(t("Agent Rerun Translated", count=len(others), pages=_names([s.stem for s in others], separator)))
    if edited:
        parts.append(t("Agent Rerun Edited", count=len(edited), pages=_names([s.stem for s in edited], separator)))
    keep = {id(s) for s in done + others}
    unfinished = [s for s in states if id(s) not in keep]
    if revisable:
        parts.append(t("Agent Rerun Revise Note"))
        parts.append(t("Agent Rerun Revise Locked") if lock else t("Agent Rerun Revise Unlocked Note"))
        if unfinished:  # two choices that take different pages, maybe as many of them
            parts.append(t("Agent Rerun Pages", revise=_names([s.stem for s in revisable], separator),
                           unfinished=_names([s.stem for s in unfinished], separator)))
    parts.append(t("Agent Rerun Stop Note"))
    return {"text": "\n\n".join(parts), "unfinished": [s.path for s in unfinished],
            "all": [s.path for s in states], "revise": [s.path for s in revisable],
            "edited": len(edited), "translated": len(others), "lock": lock}


def ask_rerun(parent, t: Callable[..., str], files: Sequence[str],
              states=None, lock: bool = True) -> Optional[Tuple[List[str], bool]]:
    """(the files to process, whether to check them again) — all of them, only the ones not done, or
    the finished ones checked again; None when the user cancels. ``lock``: the run's
    ``revise_lock_user_edits``, so the check-again button says what happens to the user's boxes."""
    if states is None:
        from manga_translator.agent.resume import previous_states
        states = previous_states(files)
    busy = [s for s in states if getattr(s, "busy", False)]
    if busy:  # another run works on the chapter: a second one would change the same pages
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(t("Agent Rerun Title"))
        box.setText(t("Agent Rerun Busy", pages=_names([s.stem for s in busy], t("Agent Result Separator"))))
        box.addButton(t("Cancel"), QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return None
    question = rerun_question(t, states, lock)
    if question is None:
        return list(files), False
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(t("Agent Rerun Title"))
    box.setText(question["text"])
    revise = None
    if question["revise"]:
        revise = box.addButton(t("Agent Rerun Revise" if lock else "Agent Rerun Revise Unlocked",
                                 count=len(question["revise"])),
                               QMessageBox.ButtonRole.ActionRole)
    only = None
    if question["unfinished"]:
        only = box.addButton(t("Agent Rerun Unfinished", count=len(question["unfinished"])),
                             QMessageBox.ButtonRole.AcceptRole)
    redo = box.addButton(t("Agent Rerun All", count=len(question["all"])), QMessageBox.ButtonRole.DestructiveRole)
    cancel = box.addButton(t("Cancel"), QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(only or cancel)
    box.setEscapeButton(cancel)
    box.exec()
    clicked = box.clickedButton()
    if revise is not None and clicked is revise:
        return question["revise"], True
    if only is not None and clicked is only:
        return question["unfinished"], False
    if clicked is redo:
        return question["all"], False
    return None
