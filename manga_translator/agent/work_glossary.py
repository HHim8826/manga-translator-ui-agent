"""Names and terms the agent learned, written back into the work's prompt file
(``translator.high_quality_prompt_path``, for example ``dict/<work>.yaml``; setting
``update_work_glossary``) after a chapter, so later chapters and the other translators use them.

* Only new entries. A term whose source is already an entry or an alias in the file is left
  alone, so the user's own entries always win.
* Only terms used in the chapter's final translations (after the proofreading), so a term the
  proofreader replaced is not saved.
* Glossary terms need one of the file's categories; ``Other`` (a sound, a one-off word) stays in
  the chapter's memory. Characters go to ``Person``, or ``Creature`` when the agent said so.
* A YAML file gets the new entries inserted as lines, in the layout of its existing entries, so its
  comments and formatting stay. The result is parsed again and must equal the old content plus the
  new entries, otherwise nothing is written. A JSON file is rewritten with two-space indentation.
* ``<file>.bak`` is the file as it was before the agent first changed it; it is never replaced.
  Each change also keeps the version before it in ``<folder>/agent/backups/`` (the last
  ``BACKUPS_KEPT``; a sub folder, so the app does not list them as prompt files).
"""
from __future__ import annotations

import codecs
import copy
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .store import AgentMemory, normalize

logger = logging.getLogger("manga_translator.agent")

WORK_CATEGORIES = ("Person", "Location", "Org", "Item", "Skill", "Creature")
"""The glossary sections of a prompt file, as the prompt editor shows them."""

_GLOSSARY = re.compile(r"^glossary\s*:\s*(#.*)?$")
_KEY = re.compile(r"""^\s*(["']?)([^"':#]+)\1\s*:(.*)$""")


@dataclass
class NewTerm:
    category: str
    source: str
    target: str
    description: str = ""

    def entry(self) -> Dict[str, Any]:
        return {"original": self.source,
                "aliases": [{"original": self.source,
                             "translations": [{"text": self.target, "condition": ""}]}],
                "description": self.description, "overwrite": False}

    def brief(self) -> Dict[str, str]:
        return {"category": self.category, "source": self.source, "target": self.target}


def known_sources(data: Dict[str, Any]) -> set:
    """Every entry and alias source in the file, in the loose search form."""
    glossary = data.get("glossary")
    groups = glossary.values() if isinstance(glossary, dict) else [glossary] if isinstance(glossary, list) else []
    found = set()
    for items in groups:
        for item in items or []:
            if not isinstance(item, dict):
                continue
            aliases = [a for a in item.get("aliases") or [] if isinstance(a, dict)]
            for value in [item.get("original")] + [a.get("original") for a in aliases]:
                if value:
                    found.add(normalize(str(value)))
    return found


BACKUPS_KEPT = 5

# the note on each added entry, in the target language: (origin, note with origin)
_ORIGIN = {
    "CHT": ("AI agent 自動加入：{where}", "{note}（{origin}）"),
    "CHS": ("AI agent 自动加入：{where}", "{note}（{origin}）"),
    "JPN": ("AI agent が自動追加：{where}", "{note}（{origin}）"),
    "KOR": ("AI agent 자동 추가: {where}", "{note} ({origin})"),
    "": ("added by the AI agent: {where}", "{note} ({origin})"),
}


def candidates(memory: AgentMemory, data: Dict[str, Any], translations: Iterable[str],
               chapter: str, target_lang: str = "CHT") -> Tuple[List[NewTerm], Dict[str, List[str]]]:
    """The terms to add, and the others by reason: ``known`` (already in the file), ``other``
    (not a name or term of the work) and ``unused`` (not in the final translations)."""
    origin_text, with_note = _ORIGIN.get((target_lang or "").upper(), _ORIGIN[""])
    known = known_sources(data)
    text = "\n".join(t for t in translations if t)
    terms: List[NewTerm] = []
    skipped: Dict[str, List[str]] = {"known": [], "other": [], "unused": []}
    seen = set()

    def consider(source: str, target: str, category: str, note: str, page: str) -> None:
        source, target, key = (source or "").strip(), (target or "").strip(), normalize(source)
        if not source or not target or key in seen:
            return
        seen.add(key)
        if key in known:
            skipped["known"].append(source)
        elif category not in WORK_CATEGORIES:
            skipped["other"].append(source)
        elif target not in text:
            skipped["unused"].append(source)
        else:
            origin = origin_text.format(where=chapter + (f" p.{page}" if page else ""))
            note = (note or "").strip()
            terms.append(NewTerm(category, source, target, with_note.format(note=note, origin=origin)
                                 if note else origin))

    for name, info in memory.characters.items():
        category = info.get("category") if info.get("category") in ("Person", "Creature") else "Person"
        consider(name, info.get("target_name", ""), category, info.get("notes", ""), info.get("page", ""))
    for entry in memory.glossary:
        consider(entry.get("source", ""), entry.get("target", ""), entry.get("category", ""),
                 entry.get("note", ""), entry.get("page", ""))
    return terms, skipped


# --- YAML as lines ------------------------------------------------------------------
def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _blank(line: str) -> bool:
    return not line.strip()


def _block_end(lines: List[str], header: int, indent: int) -> int:
    """First line after ``header`` that no longer belongs to its block."""
    for i in range(header + 1, len(lines)):
        line = lines[i]
        if _blank(line):
            continue
        if _indent(line) < indent or (_indent(line) == indent and not line.lstrip().startswith("- ")):
            return i
    return len(lines)


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)  # a JSON string is a valid YAML double-quoted scalar


def _entry_lines(term: NewTerm, indent: int) -> List[str]:
    p = " " * indent
    return [f"{p}- original: {_quote(term.source)}",
            f"{p}  aliases:",
            f"{p}    - original: {_quote(term.source)}",
            f"{p}      translations:",
            f"{p}        - text: {_quote(term.target)}",
            f'{p}          condition: ""',
            f"{p}  description: {_quote(term.description)}",
            f"{p}  overwrite: false"]


def _entries(terms: List[NewTerm], indent: int, first_blank: bool) -> List[str]:
    out: List[str] = []
    for term in terms:
        if out or first_blank:
            out.append("")
        out += _entry_lines(term, indent)
    return out


def insert_yaml(text: str, terms: List[NewTerm]) -> str:
    """``text`` (``\\n`` line ends, no trailing newline) with ``terms`` added to their categories
    of the ``glossary`` block. Raises ValueError for a layout it does not handle."""
    lines = text.split("\n") if text else []
    start = next((i for i, line in enumerate(lines) if _GLOSSARY.match(line)), None)
    if start is None:
        lines += ([""] if lines and not _blank(lines[-1]) else []) + ["glossary:"]
        start = len(lines) - 1
    for category in WORK_CATEGORIES:
        group = [t for t in terms if t.category == category]
        if not group:
            continue
        end = _block_end(lines, start, 0)
        body = [i for i in range(start + 1, end) if not _blank(lines[i]) and not lines[i].lstrip().startswith("#")]
        child = _indent(lines[body[0]]) if body else 2
        if child == 0:
            raise ValueError("the glossary block is not a mapping of categories")
        header = None
        for i in body:
            match = _KEY.match(lines[i]) if _indent(lines[i]) == child else None
            if match and match.group(2).strip() == category:
                header = i
                break
        if header is None:  # a new category at the end of the block
            last = body[-1] if body else start
            new = ([""] if body else []) + [" " * child + f"{category}:"] + _entries(group, child + 2, False)
            lines[last + 1:last + 1] = new
            continue
        value = _KEY.match(lines[header]).group(3).split("#", 1)[0].strip()
        if value not in ("", "[]", "null", "~"):
            raise ValueError(f"the {category} list is written inline")
        if value:
            lines[header] = " " * child + f"{category}:"
        block_end = _block_end(lines, header, child)
        content = [i for i in range(header + 1, block_end) if not _blank(lines[i])]
        items = [i for i in content if lines[i].lstrip().startswith("- ")]
        indent = _indent(lines[items[0]]) if items else child + 2
        last = content[-1] if content else header
        new = _entries(group, indent, bool(content))
        if last + 1 < len(lines) and not _blank(lines[last + 1]):
            new.append("")  # keep the next category apart, as the file does
        lines[last + 1:last + 1] = new
    return "\n".join(lines)


def _expected(data: Dict[str, Any], terms: List[NewTerm]) -> Dict[str, Any]:
    out = copy.deepcopy(data) if isinstance(data, dict) else {}
    glossary = out.get("glossary")
    if glossary is None:
        glossary = out["glossary"] = {}
    for term in terms:
        glossary[term.category] = list(glossary.get(term.category) or []) + [term.entry()]
    return out


# --- the file ---------------------------------------------------------------------
def read_prompt_file(path: str) -> Dict[str, Any]:
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8-sig")
    if os.path.splitext(path)[1].lower() == ".json":
        data = json.loads(text) if text.strip() else {}
    else:
        import yaml
        data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("the prompt file is not a mapping")
    if data.get("glossary") is not None and not isinstance(data["glossary"], dict):
        raise ValueError("the glossary of the prompt file is not divided into categories")
    return data


def write_terms(path: str, terms: List[NewTerm]) -> None:
    """Add ``terms`` to the prompt file at ``path``; the old file is kept as ``<path>.bak``."""
    with open(path, "rb") as fh:
        raw = fh.read()
    bom = raw.startswith(codecs.BOM_UTF8)
    text = raw.decode("utf-8-sig")
    data = read_prompt_file(path)
    expected = _expected(data, terms)
    if os.path.splitext(path)[1].lower() == ".json":
        new_text = json.dumps(expected, ensure_ascii=False, indent=2) + "\n"
    else:
        import yaml
        newline = "\r\n" if "\r\n" in text else "\n"
        body = text.replace("\r\n", "\n")
        stripped = body.rstrip("\n")
        new_text = (insert_yaml(stripped, terms) + body[len(stripped):]).replace("\n", newline)
        if yaml.safe_load(new_text) != expected:
            raise ValueError("the edited file would not read back as the old content plus the new terms")
    backup(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write((codecs.BOM_UTF8 if bom else b"") + new_text.encode("utf-8"))
    os.replace(tmp, path)


def backup(path: str) -> str:
    """Keep ``path`` before a change: ``<path>.bak`` once (the user's own version), and a dated copy
    in ``agent/backups/`` next to it, of which the last ``BACKUPS_KEPT`` stay. Returns the copy."""
    first = path + ".bak"
    if not os.path.exists(first):
        shutil.copy2(path, first)
    folder = os.path.join(os.path.dirname(os.path.abspath(path)), "agent", "backups")
    os.makedirs(folder, exist_ok=True)
    name, ext = os.path.splitext(os.path.basename(path))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target, n = os.path.join(folder, f"{name}.{stamp}{ext}"), 1
    while os.path.exists(target):  # two changes in one second
        target, n = os.path.join(folder, f"{name}.{stamp}-{n:02d}{ext}"), n + 1
    shutil.copy2(path, target)
    dated = re.compile(re.escape(name) + r"\.\d{8}-\d{6}(?:-\d{2})?" + re.escape(ext) + "$")
    copies = sorted(f for f in os.listdir(folder) if dated.match(f))
    for old in copies[:-BACKUPS_KEPT]:
        os.remove(os.path.join(folder, old))
    return target


def update_work_glossary(path: str, memory: AgentMemory, translations: Iterable[str],
                         chapter: str, target_lang: str = "CHT") -> Dict[str, Any]:
    """Save the chapter's new terms; returns what was added and what was left out (by reason)."""
    data = read_prompt_file(path)
    terms, skipped = candidates(memory, data, translations, chapter, target_lang)
    if terms:
        write_terms(path, terms)
    return {"file": os.path.basename(path), "added": [t.brief() for t in terms],
            "skipped": {reason: names for reason, names in skipped.items() if names}}
