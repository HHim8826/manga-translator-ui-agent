"""State files of the agent workflow and its long-term memory.

Layout (design doc, "資料格式")::

    manga_translator_work/agent/
        memory/glossary.json     terms, each with the page it came from
        memory/characters.json   how characters are named and how they talk
        memory/story.md          running story summary
        pages/<stem>.agent.json  per-page sidecar: reasons, flags, agent readings
        runs/<run_id>.jsonl      event log (see ``events``)

The state files are the source of truth; the conversation can be compacted or
rebuilt from them at any time.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from typing import Any, Dict, List, Optional


def page_stem(image_path: str) -> str:
    """The name the agent knows a page by (its record, events and lists): its stem, or its file name
    when another image of its folder has the same stem (``path_manager.page_name``)."""
    from ..utils.path_manager import page_name

    return page_name(image_path)

SIDECAR_VERSION = 1


def write_json_atomic(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def normalize(text: str) -> str:
    """Loose form for searching: NFKC, lower case, no spaces."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "")).lower()


class AgentStore:
    """Paths and file I/O for one chapter's agent data."""

    def __init__(self, work_dir: str, memory_dir: str = ""):
        self.work_dir = work_dir
        self.agent_dir = os.path.join(work_dir, "agent")
        self.memory_dir = memory_dir or os.path.join(self.agent_dir, "memory")
        self.pages_dir = os.path.join(self.agent_dir, "pages")
        self.runs_dir = os.path.join(self.agent_dir, "runs")

    def sidecar_path(self, stem: str) -> str:
        return os.path.join(self.pages_dir, f"{stem}.agent.json")

    def load_sidecar(self, stem: str) -> Optional[dict]:
        return _read_json(self.sidecar_path(stem), None)

    def save_sidecar(self, stem: str, data: dict) -> None:
        write_json_atomic(self.sidecar_path(stem), {"version": SIDECAR_VERSION, **data})

    def candidates_dir(self, stem: str) -> str:
        return os.path.join(self.agent_dir, "candidates", stem)

    def run_log_path(self, run_id: str) -> str:
        return os.path.join(self.runs_dir, f"{run_id}.jsonl")

    def latest_run_log(self) -> Optional[str]:
        if not os.path.isdir(self.runs_dir):
            return None
        logs = sorted(name for name in os.listdir(self.runs_dir) if name.endswith(".jsonl"))
        return os.path.join(self.runs_dir, logs[-1]) if logs else None

    def memory(self) -> "AgentMemory":
        return AgentMemory(self.memory_dir)


class AgentMemory:
    """Glossary, character table and story summary, saved on every change."""

    def __init__(self, memory_dir: str):
        self.dir = memory_dir
        self._glossary_path = os.path.join(memory_dir, "glossary.json")
        self._characters_path = os.path.join(memory_dir, "characters.json")
        self._story_path = os.path.join(memory_dir, "story.md")
        self._styles_path = os.path.join(memory_dir, "font_styles.json")
        self.glossary: List[Dict[str, str]] = _read_json(self._glossary_path, [])
        self.characters: Dict[str, Dict[str, str]] = _read_json(self._characters_path, {})
        self.font_styles: Dict[str, Dict[str, str]] = _read_json(self._styles_path, {})
        """Style name → {"font", "reason", "page"} (``font_styles.py``)."""
        try:
            with open(self._story_path, "r", encoding="utf-8") as fh:
                self.story = fh.read().strip()  # as story_update wrote it
        except FileNotFoundError:
            self.story = ""

    # glossary
    def glossary_search(self, query: str, limit: int = 30) -> List[Dict[str, str]]:
        q = normalize(query)
        if not q:
            return self.glossary[:limit]
        hits = [e for e in self.glossary
                if q in normalize(e.get("source", "")) or q in normalize(e.get("target", ""))
                or q in normalize(e.get("note", ""))]
        return hits[:limit]

    def glossary_upsert(self, source: str, target: str, note: str = "", page: str = "", category: str = "") -> str:
        """``category`` is the section of the work's prompt file the term belongs in (``Person``,
        ``Item``, ...); empty for words that are not names or terms of the work."""
        source, target, category = source.strip(), target.strip(), (category or "").strip()
        if not source or not target:
            raise ValueError("source and target are required")
        for entry in self.glossary:
            if normalize(entry.get("source", "")) == normalize(source):
                entry.update({"target": target, "note": note or entry.get("note", ""), "updated_page": page})
                if category:
                    entry["category"] = category
                write_json_atomic(self._glossary_path, self.glossary)
                return "updated"
        entry = {"source": source, "target": target, "note": note, "page": page}
        if category:
            entry["category"] = category
        self.glossary.append(entry)
        write_json_atomic(self._glossary_path, self.glossary)
        return "added"

    # characters
    def character_get(self, name: str = "") -> Dict[str, Dict[str, str]]:
        if not name:
            return dict(self.characters)
        q = normalize(name)
        return {k: v for k, v in self.characters.items()
                if q in normalize(k) or q in normalize(v.get("target_name", ""))
                or q in normalize(v.get("reading", ""))}

    def character_upsert(self, name: str, fields: Dict[str, str], page: str = "") -> str:
        name = name.strip()
        if not name:
            raise ValueError("name is required")
        clean = {k: str(v).strip() for k, v in fields.items() if v not in (None, "")}
        status = "updated" if name in self.characters else "added"
        entry = self.characters.setdefault(name, {"page": page})
        entry.update(clean)
        if status == "updated":
            entry["updated_page"] = page
        write_json_atomic(self._characters_path, self.characters)
        return status

    # font styles
    def font_style_set(self, name: str, font: str, reason: str, page: str = "") -> str:
        status = "updated" if name in self.font_styles else "added"
        self.font_styles[name] = {"font": font, "reason": reason.strip(), "page": page}
        write_json_atomic(self._styles_path, self.font_styles)
        return status

    def font_style_font(self, name: str, font: str) -> None:
        """Give a style another font and keep the rest of its entry (where it came from, its reason)."""
        self.font_styles[name] = {**self.font_styles[name], "font": font}
        write_json_atomic(self._styles_path, self.font_styles)

    def font_styles_seed(self, styles: Dict[str, Dict[str, str]]) -> List[str]:
        """Styles of earlier chapters that this chapter does not have yet; returns their names."""
        added = [name for name in styles if name not in self.font_styles]
        for name in added:
            entry = styles[name]
            self.font_styles[name] = {"font": entry.get("font", ""), "reason": entry.get("reason", ""),
                                      "page": "", "from": entry.get("chapter", "") or "work"}
        if added:
            write_json_atomic(self._styles_path, self.font_styles)
        return added

    # story
    def story_update(self, summary: str) -> None:
        self.story = summary.strip()
        os.makedirs(self.dir, exist_ok=True)
        with open(self._story_path, "w", encoding="utf-8") as fh:
            fh.write(self.story + "\n")

    def replace(self, glossary: List[Dict[str, str]], characters: Dict[str, Dict[str, str]], story: str) -> None:
        """Set and save the glossary, the characters and the story at once (``work_memory.py``)."""
        self.glossary = glossary
        self.characters = characters
        write_json_atomic(self._glossary_path, self.glossary)
        write_json_atomic(self._characters_path, self.characters)
        if story or self.story:
            self.story_update(story)

    def counts(self) -> Dict[str, int]:
        return {"glossary": len(self.glossary), "characters": len(self.characters),
                "story_chars": len(self.story)}

    def snapshot_text(self, max_glossary: int = 200) -> str:
        """Memory as text for the start of a conversation."""
        parts = []
        if self.glossary:
            lines = [f"- {e['source']} → {e['target']}" + (f" ({e['note']})" if e.get("note") else "")
                     for e in self.glossary[:max_glossary]]
            more = len(self.glossary) - max_glossary
            parts.append("Glossary:\n" + "\n".join(lines) + (f"\n(+{more} more; use glossary_search)" if more > 0 else ""))
        if self.characters:
            lines = []
            for name, info in self.characters.items():
                details = ", ".join(f"{k}: {v}" for k, v in info.items() if k not in ("page", "updated_page", "chapter"))
                lines.append(f"- {name}: {details}")
            parts.append("Characters:\n" + "\n".join(lines))
        if self.story:
            parts.append("Story so far:\n" + self.story)
        return "\n\n".join(parts) if parts else "Memory is empty: this is the first chapter seen."
