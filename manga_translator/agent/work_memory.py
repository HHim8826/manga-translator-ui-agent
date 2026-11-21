"""What the chapters of one work share: the glossary, the characters and the story so far.

Each chapter keeps its own memory (``manga_translator_work/agent/memory``), so a rerun of an earlier
chapter sees what it knew then, not the story of the chapters after it. The work's memory sits next
to its prompt file (``translator.high_quality_prompt_path``), in ``<folder>/agent/<name>.memory/``:

* the first run of a chapter (its own memory is empty) starts with what the chapters before it
  learned (``seed_chapter``); what it was given is kept in its memory folder (``work_seed.json``);
* after each run of a chapter, what the chapter added or changed compared with what it was given is
  kept in ``chapters/<chapter>.json`` (``save_chapter``), and the work's memory is put together
  again from those, in chapter order: a later chapter's term, character detail or story wins over
  an earlier one's, whichever was run last. Rerunning chapter 1 after chapter 2 changes what chapter
  1 says, not what chapter 2 says.

A work memory saved before the chapters were kept apart becomes the bottom layer (``base.json``),
and a chapter that was shared before it had a ``work_seed.json`` counts only what its runs after
that change. The first time, the work's memory is put together from the chapters next to this one
that come before it and were run with the same prompt file (``bootstrap``), so a work that was
started before this existed carries on. Font styles are shared through the work's own file
(``font_styles.py``). With ``memory_dir`` set, every chapter uses that one folder instead and
nothing is copied.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional

from .store import AgentMemory, _read_json, normalize, write_json_atomic

logger = logging.getLogger("manga_translator.agent")

META = "chapters.json"
SEED = "work_seed.json"  # in a chapter's memory folder: what the work gave it
BASE = "base.json"  # in the work's folder: the memory from before the chapters were kept apart
LAYERS = "chapters"  # in the work's folder: what each chapter added or changed
BOOKKEEPING = ("page", "updated_page", "chapter")


def work_memory_dir(prompt_path: str) -> str:
    """Where the work's memory lives, next to its prompt file; empty without one."""
    if not prompt_path:
        return ""
    folder, name = os.path.split(os.path.abspath(prompt_path))
    return os.path.join(folder, "agent", os.path.splitext(name)[0] + ".memory")


def _empty(memory) -> bool:
    return not (memory.glossary or memory.characters or memory.story)


def _counts(memory) -> Dict[str, int]:
    return {"glossary": len(memory.glossary), "characters": len(memory.characters), "story": len(memory.story)}


def chapter_order(name: str) -> List[object]:
    """Sort key of a chapter folder: numbers by value, full-width digits too (``第３３話`` < ``第１００話``)."""
    text = unicodedata.normalize("NFKC", name).lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


class _Layer:
    """A memory as plain data: a chapter's changes, the seed it was given, or the merged result."""

    def __init__(self, glossary=None, characters=None, story: str = ""):
        self.glossary: List[Dict[str, Any]] = [dict(e) for e in glossary or [] if isinstance(e, dict)]
        self.characters: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in (characters or {}).items()
                                                      if isinstance(v, dict)}
        self.story = str(story or "")

    @classmethod
    def of(cls, memory) -> "_Layer":
        return cls(memory.glossary, memory.characters, memory.story)

    @classmethod
    def read(cls, path: str) -> Optional["_Layer"]:
        data = _read_json(path, None)
        if not isinstance(data, dict):
            return None
        return cls(data.get("glossary"), data.get("characters"), data.get("story"))

    def data(self, **extra) -> dict:
        return {**extra, "glossary": self.glossary, "characters": self.characters, "story": self.story}

    def write(self, path: str, **extra) -> None:
        write_json_atomic(path, self.data(**extra))


def _changes(memory, seed: _Layer) -> _Layer:
    """What a chapter's memory added or changed compared with what the work gave it: whole terms,
    the character details that differ, and its story when it wrote one."""
    given = {normalize(e.get("source", "")): e for e in seed.glossary}
    glossary = [dict(e) for e in memory.glossary
                if normalize(e.get("source", "")) and given.get(normalize(e.get("source", ""))) != e]
    characters = {}
    for person, info in memory.characters.items():
        before = seed.characters.get(person)
        if before is None:
            characters[person] = dict(info)
            continue
        changed = {k: v for k, v in info.items() if before.get(k) != v}
        if any(k not in BOOKKEEPING for k in changed):
            characters[person] = changed
    story = memory.story if memory.story and memory.story != seed.story else ""
    return _Layer(glossary, characters, story)


def _apply(result: _Layer, layer: _Layer, name: str) -> None:
    where = {normalize(e.get("source", "")): i for i, e in enumerate(result.glossary)}
    for entry in layer.glossary:
        entry = dict(entry)
        key = normalize(entry.get("source", ""))
        if not key:
            continue
        if name:
            entry.setdefault("chapter", name)  # the chapter the term was found in, for the prompt file's notes
        if key in where:
            result.glossary[where[key]] = entry
        else:
            where[key] = len(result.glossary)
            result.glossary.append(entry)
    for person, info in layer.characters.items():
        info = dict(info)
        if name and person not in result.characters:
            info.setdefault("chapter", name)
        result.characters[person] = {**result.characters.get(person, {}), **info}
    if layer.story:
        result.story = layer.story


def _where(folder: str) -> str:
    return os.path.normcase(os.path.abspath(folder)) if folder else ""


def _key(name: str, folder: str) -> tuple:
    """Chapter order; two chapters with the same folder name in different places by their path."""
    return chapter_order(name), chapter_order(folder)


def _same(a: tuple, b: tuple) -> bool:
    """Whether two (name, folder) are one chapter; without a folder the name decides."""
    return a[0] == b[0] and (not a[1] or not b[1] or a[1] == b[1])


def _layers(work_dir: str) -> List[tuple]:
    """(chapter name, its folder, its changes) for every chapter of the work, in chapter order."""
    folder = os.path.join(work_dir, LAYERS)
    found = []
    try:
        names = sorted(n for n in os.listdir(folder) if n.endswith(".json"))
    except OSError:
        return []
    for file_name in names:
        data = _read_json(os.path.join(folder, file_name), None)
        if not isinstance(data, dict):
            continue
        name = str(data.get("name") or os.path.splitext(file_name)[0])
        found.append((name, str(data.get("folder") or ""),
                      _Layer(data.get("glossary"), data.get("characters"), data.get("story"))))
    return sorted(found, key=lambda item: _key(item[0], item[1]))


def _built(work_dir: str, before: Optional[tuple] = None) -> tuple:
    """The work's memory from its layers; with ``before`` (name, folder), only the chapters that
    come before that one. Returns (memory, the chapter its story is from)."""
    result = _Layer.read(os.path.join(work_dir, BASE)) or _Layer()
    story_from = ""
    for name, folder, layer in _layers(work_dir):
        if before is not None and (_same((name, folder), before) or _key(name, folder) >= _key(*before)):
            continue
        _apply(result, layer, name)
        if layer.story:
            story_from = name
    return result, story_from


def _layer_path(work_dir: str, name: str, folder: str) -> str:
    tag = f".{hashlib.sha1(folder.encode('utf-8')).hexdigest()[:8]}" if folder else ""
    return os.path.join(work_dir, LAYERS, f"{name}{tag}.json")


def _migrate(work_dir: str) -> None:
    """A work memory saved before the chapters were kept apart becomes the bottom layer."""
    if not os.path.isdir(work_dir) or os.path.isdir(os.path.join(work_dir, LAYERS)) \
            or os.path.exists(os.path.join(work_dir, BASE)):
        return
    work = AgentMemory(work_dir)
    if not _empty(work):
        _Layer.of(work).write(os.path.join(work_dir, BASE))


def seed_chapter(chapter: AgentMemory, work_dir: str, name: str = "", folder: str = "") -> Dict[str, int]:
    """A chapter's first run starts with what the chapters before it learned; returns what was
    copied (nothing when the chapter already has a memory of its own, or the work has none). The
    copy is kept in the chapter's memory folder, so its own changes can be told apart later."""
    if not work_dir:
        return {}
    seed_path = os.path.join(chapter.dir, SEED)
    if not _empty(chapter):
        if not os.path.exists(seed_path):
            # a chapter from before the seeds were kept: what was shared already is in base.json
            meta = _read_json(os.path.join(work_dir, META), {})
            shared = name in (meta.get("chapters") or {}) if isinstance(meta, dict) else False
            (_Layer.of(chapter) if shared else _Layer()).write(seed_path)
        return {}
    _migrate(work_dir)
    work, _ = _built(work_dir, before=(name, _where(folder)) if name else None) \
        if os.path.isdir(work_dir) else (_Layer(), "")
    work.write(seed_path)
    if _empty(work):
        return {}
    chapter.replace(glossary=[dict(e) for e in work.glossary],
                    characters={person: dict(info) for person, info in work.characters.items()}, story=work.story)
    return _counts(work)


def save_chapter(chapter: AgentMemory, work_dir: str, name: str, folder: str = "") -> Dict[str, int]:
    """Keep what a finished chapter (its folder's name, and the folder) added or changed and put
    the work's memory together again; returns what changed in it."""
    if not work_dir:
        return {}
    _migrate(work_dir)
    work = AgentMemory(work_dir)
    old = _Layer.of(work)
    seed = _Layer.read(os.path.join(chapter.dir, SEED)) or _Layer()
    folder = _where(folder)
    _changes(chapter, seed).write(_layer_path(work_dir, name, folder), name=name, folder=folder)
    merged, story_from = _built(work_dir)
    work.replace(glossary=merged.glossary, characters=merged.characters, story=merged.story)

    before = {normalize(e.get("source", "")): e for e in old.glossary}
    added = sum(1 for e in merged.glossary if normalize(e.get("source", "")) not in before)
    updated = sum(1 for e in merged.glossary
                  if normalize(e.get("source", "")) in before and before[normalize(e.get("source", ""))] != e)
    meta_path = os.path.join(work_dir, META)
    meta = _read_json(meta_path, {})
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("chapters", {})[name] = time.strftime("%Y-%m-%d %H:%M:%S")
    if story_from:
        meta["story_from"] = story_from
    write_json_atomic(meta_path, meta)
    return {"glossary_added": added, "glossary_updated": updated,
            "characters_added": sum(1 for person in merged.characters if person not in old.characters),
            "story": len(merged.story)}


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
        save_chapter(memory, work_dir, name, os.path.join(parent, name))  # no seed: all of it is its own
        merged.append(name)
    return merged
