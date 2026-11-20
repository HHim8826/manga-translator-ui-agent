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
        self.glossary: List[Dict[str, str]] = _read_json(self._glossary_path, [])
        self.characters: Dict[str, Dict[str, str]] = _read_json(self._characters_path, {})
        try:
            with open(self._story_path, "r", encoding="utf-8") as fh:
                self.story = fh.read()
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

    def glossary_upsert(self, source: str, target: str, note: str = "", page: str = "") -> str:
        source, target = source.strip(), target.strip()
        if not source or not target:
            raise ValueError("source and target are required")
        for entry in self.glossary:
            if normalize(entry.get("source", "")) == normalize(source):
                entry.update({"target": target, "note": note or entry.get("note", ""), "updated_page": page})
                write_json_atomic(self._glossary_path, self.glossary)
                return "updated"
        self.glossary.append({"source": source, "target": target, "note": note, "page": page})
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

    # story
    def story_update(self, summary: str) -> None:
        self.story = summary.strip()
        os.makedirs(self.dir, exist_ok=True)
        with open(self._story_path, "w", encoding="utf-8") as fh:
            fh.write(self.story + "\n")

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
                details = ", ".join(f"{k}: {v}" for k, v in info.items() if k not in ("page", "updated_page"))
                lines.append(f"- {name}: {details}")
            parts.append("Characters:\n" + "\n".join(lines))
        if self.story:
            parts.append("Story so far:\n" + self.story)
        return "\n\n".join(parts) if parts else "Memory is empty: this is the first chapter seen."
