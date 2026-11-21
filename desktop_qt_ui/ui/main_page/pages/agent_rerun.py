"""Asks before an AI Agent run redoes pages an earlier run already did (``manga_translator.agent.resume``).

The preprocess writes every page's JSON again, so a rerun replaces the agent's translations and the
changes made in the editor. When pages of the run already have an agent result, the user chooses:
only the pages that are not done, every page, or nothing.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from PyQt6.QtWidgets import QMessageBox

MAX_NAMES = 8


def _names(stems: Sequence[str], separator: str) -> str:
    shown = separator.join(stems[:MAX_NAMES])
    return shown + ("…" if len(stems) > MAX_NAMES else "")


def rerun_question(t: Callable[..., str], states) -> Optional[dict]:
    """Texts and page lists of the question; None when no page has an agent result."""
    seen = [s for s in states if s.status]
    if not seen:
        return None
    done = [s for s in seen if s.status == "done"]
    edited = [s for s in seen if s.edited]
    text = t("Agent Rerun Text", seen=len(seen), pages=len(states), done=len(done))
    if edited:
        text += "\n\n" + t("Agent Rerun Edited", count=len(edited),
                           pages=_names([s.stem for s in edited], t("Agent Result Separator")))
    return {"text": text, "unfinished": [s.path for s in states if s.status != "done"],
            "all": [s.path for s in states], "edited": len(edited)}


def ask_rerun(parent, t: Callable[..., str], files: Sequence[str],
              states=None) -> Optional[List[str]]:
    """The files to process: all of them, only the ones not done, or None when the user cancels."""
    if states is None:
        from manga_translator.agent.resume import previous_states
        states = previous_states(files)
    question = rerun_question(t, states)
    if question is None:
        return list(files)
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(t("Agent Rerun Title"))
    box.setText(question["text"])
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
    if only is not None and clicked is only:
        return question["unfinished"]
    if clicked is redo:
        return question["all"]
    return None
