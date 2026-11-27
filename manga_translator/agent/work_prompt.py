"""The work's own prompt file (translator setting ``high_quality_prompt_path``, for example
``dict/<work>.yaml``) for the agent: its free-text ``system_prompt`` and its glossary go into
the agent's system prompt, and the glossary terms feed the program checks.

New terms the agent finds go to its own memory first; after a chapter, ``work_glossary`` adds the
new names and terms to this file (setting ``update_work_glossary``), never changing its entries.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("manga_translator.agent")


@dataclass
class WorkTerm:
    source: str
    targets: List[str]
    category: str = ""
    condition: str = ""


@dataclass
class WorkPrompt:
    name: str
    instructions: str = ""
    text: str = ""
    """What goes into the agent's system prompt."""
    terms: List[WorkTerm] = field(default_factory=list)
    path: str = ""
    """Where the file was found."""


def resolve_path(path: str) -> Optional[str]:
    """The prompt file as the translators find it: as given, or relative to the app folder."""
    if not path:
        return None
    candidates = [path]
    if not os.path.isabs(path):
        try:
            from ..utils import BASE_PATH
            candidates.append(os.path.join(BASE_PATH, path))
        except Exception:  # pragma: no cover
            pass
    for candidate in candidates:
        base, ext = os.path.splitext(candidate)
        for option in (candidate, *(base + e for e in (".yaml", ".yml", ".json") if e != ext)):
            if os.path.isfile(option):
                return option
    return None


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _translations(item: Dict[str, Any]) -> List[Dict[str, str]]:
    out = []
    for entry in item.get("translations") or []:
        if isinstance(entry, dict) and _text(entry.get("text")):
            out.append({"text": _text(entry["text"]), "condition": _text(entry.get("condition"))})
        elif isinstance(entry, str) and entry.strip():
            out.append({"text": entry.strip(), "condition": ""})
    if not out and _text(item.get("translation")):
        out.append({"text": _text(item["translation"]), "condition": ""})
    return out


def parse(data: Dict[str, Any], name: str, target_lang: str = "") -> WorkPrompt:
    from ..translators.common import _flatten_prompt_data

    prompt = WorkPrompt(name=name)
    instructions = _text(data.get("system_prompt"))
    if target_lang:
        instructions = instructions.replace("{{{target_lang}}}", target_lang).replace("{{target_lang}}", target_lang)
    prompt.instructions = instructions
    lines: List[str] = []
    glossary = data.get("glossary")
    if isinstance(glossary, dict):
        for category, items in glossary.items():
            if not isinstance(items, list) or not items:
                continue
            lines.append(f"[{category}]")
            for item in items:
                if not isinstance(item, dict):
                    continue
                aliases = [a for a in item.get("aliases") or [] if isinstance(a, dict) and _text(a.get("original"))]
                if not aliases and _text(item.get("original")):
                    aliases = [item]
                forms = []
                for alias in aliases:
                    found = _translations(alias)
                    if not found:
                        continue
                    prompt.terms.append(WorkTerm(_text(alias["original"]), [t["text"] for t in found], str(category),
                                                 "; ".join(t["condition"] for t in found if t["condition"])))
                    forms.append(_text(alias["original"]) + " → " + " / ".join(
                        t["text"] + (f" ({t['condition']})" if t["condition"] else "") for t in found))
                if not forms:
                    continue
                lines.append("- " + "; ".join(forms))
                if _text(item.get("description")):
                    lines.append("  " + _text(item["description"]))
    elif glossary:
        lines.append(_flatten_prompt_data(glossary))
    extra = {k: v for k, v in data.items() if k not in ("system_prompt", "glossary") and v}
    parts = []
    if instructions:
        parts.append(instructions)
    if lines:
        parts.append("Glossary (these translations are fixed; a condition says when that form applies):\n"
                     + "\n".join(lines))
    if extra:
        parts.append(_flatten_prompt_data(extra))
    prompt.text = "\n\n".join(parts)
    return prompt


def load_work_prompt(path: str, target_lang: str = "") -> Optional[WorkPrompt]:
    resolved = resolve_path(path)
    if resolved is None:
        if path:
            logger.warning(f"agent: the work prompt file {path} was not found")
        return None
    from ..translators.prompt_loader import load_prompt_file

    try:
        with open(resolved, "rb") as fh:
            blank = not fh.read().decode("utf-8-sig", "replace").strip()
        data = {} if blank else load_prompt_file(resolved)  # an empty .json logged a parse error
        if data is None:  # an empty file (blank, or {}) is a work prompt with nothing in it yet
            from .work_glossary import read_prompt_file

            data = read_prompt_file(resolved)
    except Exception as exc:  # a broken file must not stop the run
        logger.warning(f"agent: cannot read the work prompt file {resolved}: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    prompt = parse(data, os.path.basename(resolved), target_lang)
    prompt.path = resolved
    # also with no text: the new terms, the fonts of the work and its memory are kept in and next to
    # it; None turned them off for a new, empty file (review 2026-11-28 B17)
    return prompt
