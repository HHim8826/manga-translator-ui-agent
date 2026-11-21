"""What the chapters of one work share: the glossary, the characters and the story so far.

Each chapter keeps its own memory (``manga_translator_work/agent/memory``), so a rerun of an earlier
chapter sees what it knew then, not the story of the chapters after it. The work's memory sits next
to its prompt file (``translator.high_quality_prompt_path``), in ``<folder>/agent/<name>.memory/``:

* the first run of a chapter (its own memory is empty) starts with a copy of it (``seed_chapter``);
* after each chapter, its memory is merged into it (``save_chapter``): its terms and characters are
  added or replace the older ones, its story summary replaces the work's.

The first time, the work's memory is put together from the chapters next to this one that come
before it and were run with the same prompt file (``bootstrap``), so a work that was started before
this existed carries on. Font styles are shared through the work's own file (``font_styles.py``).
With ``memory_dir`` set, every chapter uses that one folder instead and nothing is copied.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional

from .store import AgentMemory, _read_json, normalize, write_json_atomic

logger = logging.getLogger("manga_translator.agent")

META = "chapters.json"


def work_memory_dir(prompt_path: str) -> str:
    """Where the work's memory lives, next to its prompt file; empty without one."""
    if not prompt_path:
        return ""
    folder, name = os.path.split(os.path.abspath(prompt_path))
    return os.path.join(folder, "agent", os.path.splitext(name)[0] + ".memory")


def _empty(memory: AgentMemory) -> bool:
    return not (memory.glossary or memory.characters or memory.story)


def _counts(memory: AgentMemory) -> Dict[str, int]:
    return {"glossary": len(memory.glossary), "characters": len(memory.characters), "story": len(memory.story)}


def chapter_order(name: str) -> List[object]:
    """Sort key of a chapter folder: numbers by value, full-width digits too (``第３３話`` < ``第１００話``)."""
    text = unicodedata.normalize("NFKC", name).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def seed_chapter(chapter: AgentMemory, work_dir: str) -> Dict[str, int]:
    """A chapter's first run starts with the work's memory; returns what was copied (nothing when
    the chapter already has a memory of its own, or the work has none)."""
    if not work_dir or not _empty(chapter) or not os.path.isdir(work_dir):
        return {}
    work = AgentMemory(work_dir)
    if _empty(work):
        return {}
    chapter.replace(glossary=[dict(e) for e in work.glossary],
                    characters={name: dict(info) for name, info in work.characters.items()}, story=work.story)
    return _counts(work)


def _merge(work: AgentMemory, chapter: AgentMemory, name: str) -> Dict[str, int]:
    glossary = [dict(e) for e in work.glossary]
    where = {normalize(e.get("source", "")): i for i, e in enumerate(glossary)}
    added = updated = 0
    for entry in chapter.glossary:
        entry = dict(entry)
        entry.setdefault("chapter", name)  # the chapter the term was found in, for the prompt file's notes
        key = normalize(entry.get("source", ""))
        if not key:
            continue
        if key in where:
            if glossary[where[key]] != entry:
                glossary[where[key]] = entry
                updated += 1
        else:
            where[key] = len(glossary)
            glossary.append(entry)
            added += 1
    characters = {k: dict(v) for k, v in work.characters.items()}
    new_characters = 0
    for person, info in chapter.characters.items():
        info = dict(info)
        info.setdefault("chapter", name)
        if person not in characters:
            new_characters += 1
        characters[person] = {**characters.get(person, {}), **info}
    story = chapter.story or work.story
    work.replace(glossary=glossary, characters=characters, story=story)
    return {"glossary_added": added, "glossary_updated": updated, "characters_added": new_characters,
            "story": len(story)}


def save_chapter(chapter: AgentMemory, work_dir: str, name: str) -> Dict[str, int]:
    """Merge a finished chapter's memory into the work's; returns what changed."""
    if not work_dir:
        return {}
    work = AgentMemory(work_dir)
    changes = _merge(work, chapter, name)
    meta_path = os.path.join(work_dir, META)
    meta = _read_json(meta_path, {})
    meta.setdefault("chapters", {})[name] = time.strftime("%Y-%m-%d %H:%M:%S")
    if chapter.story:
        meta["story_from"] = name
    write_json_atomic(meta_path, meta)
    return changes


def _prompt_of_last_run(chapter_work_dir: str) -> Optional[str]:
    """The prompt file name in the ``run_start`` of the chapter's latest run log."""
    runs = os.path.join(chapter_work_dir, "agent", "runs")
    try:
        logs = sorted(n for n in os.listdir(runs) if n.endswith(".jsonl"))
    except OSError:
        return None
    for log_name in reversed(logs):
        try:
            with open(os.path.join(runs, log_name), "r", encoding="utf-8") as fh:
                first = json.loads(fh.readline() or "{}")
        except (OSError, ValueError):
            continue
        if first.get("type") == "run_start":
            prompt = first.get("work_prompt") or {}
            return prompt.get("name") if isinstance(prompt, dict) else None
    return None


def bootstrap(work_dir: str, chapter_work_dir: str, prompt_name: str) -> List[str]:
    """When the work has no memory yet: merge the memories of the chapters next to this one that
    come before it and were run with the same prompt file, in chapter order. Returns their names."""
    if not work_dir or os.path.isdir(work_dir) or not prompt_name:
        return []
    chapter_folder = os.path.dirname(os.path.abspath(chapter_work_dir))
    parent, this = os.path.split(chapter_folder)
    work_name = os.path.basename(chapter_work_dir)
    try:
        siblings = [n for n in os.listdir(parent) if os.path.isdir(os.path.join(parent, n))]
    except OSError:
        return []
    merged = []
    for name in sorted(siblings, key=chapter_order):
        if chapter_order(name) >= chapter_order(this):
            continue
        other = os.path.join(parent, name, work_name)
        memory_dir = os.path.join(other, "agent", "memory")
        if not os.path.isdir(memory_dir) or _prompt_of_last_run(other) != prompt_name:
            continue
        memory = AgentMemory(memory_dir)
        if _empty(memory):
            continue
        save_chapter(memory, work_dir, name)
        merged.append(name)
    return merged
