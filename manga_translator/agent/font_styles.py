"""Font styles: kinds of lettering (narration, handwriting, a sign ...) mapped to a font.

The translation agent defines a style once with ``set_font_style`` (a short name, a font of the
font sheet and what the original lettering looks like), then names the style in ``set_text`` for
every region with that lettering, so the same kind of text gets the same font all through the
chapter. Normal dialogue has no style and keeps the user's font.

The table is part of the chapter's memory (``font_styles.json`` in the memory folder). When the run
uses the work's prompt file, it is also kept for the work in
``<folder of the prompt file>/agent/<prompt file name>.fonts.json`` (a sub folder, so the app does
not list it as a prompt file), and the next chapter starts with the same styles.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List

from .store import _read_json, write_json_atomic

logger = logging.getLogger("manga_translator.agent")

MAX_NAME = 30


def style_name(value) -> str:
    """A style name as stored: trimmed, inner spaces collapsed, lower case for Latin letters."""
    name = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not name:
        raise ValueError("a style needs a name, like narration or handwritten")
    if len(name) > MAX_NAME:
        raise ValueError(f"style names are at most {MAX_NAME} characters")
    return name


def work_fonts_path(prompt_path: str) -> str:
    """Where the work's styles live, next to its prompt file; empty without one."""
    if not prompt_path:
        return ""
    folder, name = os.path.split(os.path.abspath(prompt_path))
    return os.path.join(folder, "agent", os.path.splitext(name)[0] + ".fonts.json")


def load_work_styles(path: str) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    try:
        data = _read_json(path, {})
    except (OSError, ValueError):
        logger.warning(f"agent: could not read the work's font styles {path}", exc_info=True)
        return {}
    styles = data.get("styles") if isinstance(data, dict) else None
    return {str(k): dict(v) for k, v in (styles or {}).items() if isinstance(v, dict) and v.get("font")}


def save_work_styles(path: str, styles: Dict[str, Dict[str, str]], chapter: str = "") -> List[str]:
    """Add or update the chapter's styles in the work's file; returns the names that changed."""
    if not path or not styles:
        return []
    saved = load_work_styles(path)
    changed = []
    for name, entry in styles.items():
        old = saved.get(name) or {}
        if old.get("font") == entry.get("font") and old.get("reason") == entry.get("reason"):
            continue
        saved[name] = {"font": entry.get("font", ""), "reason": entry.get("reason", ""),
                       "chapter": chapter or old.get("chapter", "")}
        changed.append(name)
    if changed:
        write_json_atomic(path, {"styles": saved})
    return changed


def styles_text(styles: Dict[str, Dict[str, str]], label=lambda font: font) -> str:
    """The table for the agent: one line per style."""
    if not styles:
        return "(no font styles yet)"
    return "\n".join(f"- {name}: {label(entry.get('font', ''))}"
                     + (f" ({entry['reason']})" if entry.get("reason") else "")
                     for name, entry in styles.items())
