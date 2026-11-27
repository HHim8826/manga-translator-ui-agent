"""What happened in each of the work's earlier chapters, a few sentences each (2026-11-28).

The story summary is one text the agent rewrites as it goes, about 500 characters: by chapter 47 of a
real work it told only of chapters 41-47, and a page that starts afresh knew nothing of the first 40
(a dragon met on a volcano, the ruins, a running joke about shopping). Each earlier chapter now has a
synopsis of its own, written from its translations with one text request (again when its translations
changed) and kept in the work's memory folder (``synopses.json``). The translator's system prompt and
the proofreader get them, the nearest chapters first, within ``SYNOPSES_CHARS``.

A synopsis is written when a later chapter starts, not at the end of its own run: the chapter's last
words (proofreading, the user's edits in the editor) are in it, and a chapter that is never followed
costs nothing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .store import _read_json, write_json_atomic
from .work_memory import _where, chapter_order, chapter_translations, earlier_chapters

logger = logging.getLogger("manga_translator.agent")

SYNOPSES = "synopses.json"
"""In the work's memory folder: normcased chapter folder -> {name, synopsis, fingerprint, written}."""
SYNOPSIS_CHARS = 250
"""About how long one chapter's synopsis is asked to be (characters of the target language)."""
SYNOPSES_CHARS = 6000
"""The synopses in a prompt, the nearest chapters first: about 24 chapters at 250 characters."""
CHAPTER_TEXT_CHARS = 24000
"""The chapter's text sent for its synopsis; a chapter of 16 pages is about 9,000 characters."""
PARALLEL = 4
"""Synopsis requests at a time (20 earlier chapters the first time)."""

SYNOPSIS_TOOL = {"type": "function", "function": {
    "name": "save_synopsis", "description": "Save the chapter's synopsis.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["synopsis"],
                   "properties": {"synopsis": {"type": "string"}}}}}

SYNOPSIS_PROMPT = """You write the synopsis of one chapter of a manga for the agent that translates its later chapters into {target}. You get the chapter's text page by page: each line's source and its translation, and on some pages a note on what happens there.

Write it in {target}, at most about {chars} characters: what happens in this chapter, who does what to whom and where, and how it ends; and what a later chapter may point back to: a running joke, a nickname, what something or someone was called, a promise, a place. Write the names as the translations do, and give a word that a later line may call back in its source form too, in brackets after its translation. No remarks on the translation itself. Answer only by calling save_synopsis."""


def _fingerprint(entries: Sequence[Dict[str, Any]]) -> str:
    digest = hashlib.sha1()
    for e in entries:
        digest.update(f"{e.get('page')}\t{e.get('source')}\t{e.get('translation')}\n".encode("utf-8"))
    return digest.hexdigest()[:16]


def page_notes(chapter_work_dir: str) -> Dict[str, str]:
    """The ``page_note`` of each page the agent finished (read only, like the translations)."""
    folder = os.path.join(chapter_work_dir, "agent", "pages")
    notes: Dict[str, str] = {}
    try:
        names = sorted(n for n in os.listdir(folder) if n.endswith(".agent.json"))
    except OSError:
        return notes
    for name in names:
        try:
            with open(os.path.join(folder, name), "r", encoding="utf-8") as fh:
                side = json.load(fh)
        except (OSError, ValueError):
            continue
        note = side.get("page_note") if isinstance(side, dict) else None
        if isinstance(note, str) and note.strip():
            notes[name[: -len(".agent.json")]] = note.strip()
    return notes


def chapter_text(entries: Sequence[Dict[str, Any]], notes: Optional[Dict[str, str]] = None) -> str:
    """The chapter page by page for its synopsis, cut at ``CHAPTER_TEXT_CHARS``."""
    notes = notes or {}
    lines: List[str] = []
    page = None
    for e in entries:
        if e.get("page") != page:
            page = e.get("page")
            lines.append(f"Page {page}" + (f" (what happens: {notes[page]})" if notes.get(page) else "") + ":")
        lines.append(f"- {e.get('source') or ''} → {e.get('translation') or ''}")
    text = "\n".join(lines)
    if len(text) > CHAPTER_TEXT_CHARS:
        text = text[:CHAPTER_TEXT_CHARS].rsplit("\n", 1)[0] + "\n(the rest of the chapter is left out)"
    return text


class Synopses:
    """``synopses.json`` in the work's memory folder."""

    def __init__(self, work_dir: str):
        self.path = os.path.join(work_dir, SYNOPSES)

    def load(self) -> Dict[str, dict]:
        data = _read_json(self.path, {})
        return data if isinstance(data, dict) else {}

    @staticmethod
    def find(known: Dict[str, dict], name: str, folder: str) -> Optional[dict]:
        """The entry of the chapter in ``folder``; one with its name when the folder moved."""
        entry = known.get(_where(folder))
        if isinstance(entry, dict):
            return entry
        named = [e for e in known.values() if isinstance(e, dict) and e.get("name") == name]
        return named[0] if len(named) == 1 else None

    def put(self, name: str, folder: str, synopsis: str, fingerprint: str) -> None:
        known = self.load()
        known[_where(folder)] = {"name": name, "synopsis": synopsis, "fingerprint": fingerprint,
                                 "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json_atomic(self.path, known)


async def write_synopsis(client, settings, target_lang: str, name: str, text: str) -> Tuple[str, Any]:
    """One chapter's synopsis from its text: (synopsis or '', the reply for its token counts)."""
    from .runner import language_name

    target = language_name(target_lang)
    messages = [{"role": "system", "content": SYNOPSIS_PROMPT.format(target=target, chars=SYNOPSIS_CHARS)},
                {"role": "user", "content": f"Chapter {name}:\n\n{text}"}]
    reply = await client.complete(messages, [SYNOPSIS_TOOL], "required",
                                  effort=getattr(settings, "review_reasoning_effort", ""), cache_key="synopsis")
    for call in reply.tool_calls:
        if call.name == SYNOPSIS_TOOL["function"]["name"]:
            try:
                args = json.loads(call.arguments or "{}")
            except ValueError:
                continue
            synopsis = str(args.get("synopsis") or "").strip() if isinstance(args, dict) else ""
            if synopsis:
                return synopsis, reply
    return "", reply


def prompt_text(known: Dict[str, dict], chapters: Sequence[Tuple[str, str]]) -> str:
    """The synopses of ``chapters`` (nearest first) for a prompt: the nearest within ``SYNOPSES_CHARS``,
    printed oldest first; the older ones are named."""
    shown: List[str] = []
    left: List[str] = []
    used = 0
    for name, folder in chapters:
        entry = Synopses.find(known, name, folder)
        if not entry or not entry.get("synopsis"):
            continue
        line = f"- {name}: {entry['synopsis']}"
        if left or (shown and used + len(line) > SYNOPSES_CHARS):
            left.append(name)
            continue
        shown.append(line)
        used += len(line) + 1
    if not shown:
        return ""
    text = ("What happened in the work's earlier chapters, oldest first. A line that points back to something "
            "earlier may mean one of these; search_past_translations shows how it was translated there:\n"
            + "\n".join(reversed(shown)))
    if left:
        text += "\nOlder chapters, not summarized here: " + ", ".join(sorted(left, key=chapter_order)) + "."
    return text


async def update(client, settings, target_lang: str, work_dir: str, chapter_work_dir: str, prompt_name: str,
                 log: Callable[[str], None] = lambda _m: None,
                 should_stop: Callable[[], bool] = lambda: False) -> Tuple[str, Dict[str, Any]]:
    """Write the synopses the work's earlier chapters lack (or whose translations changed since), then
    return (the text for the prompts, '' without earlier chapters; what was written, for the chapter's
    run_start, {} when nothing was to write). A failed request is logged and that chapter goes without
    a synopsis this time; a stop leaves the rest unwritten."""
    if not work_dir:
        return "", {}
    chapters = earlier_chapters(work_dir, chapter_work_dir, prompt_name)
    if not chapters:
        return "", {}
    store = Synopses(work_dir)
    known = store.load()
    work_name = os.path.basename(chapter_work_dir)
    todo = []
    info: Dict[str, Any] = {}
    for name, folder in chapters:
        entries = chapter_translations(os.path.join(folder, work_name), name)
        if not entries:
            continue
        fingerprint = _fingerprint(entries)
        entry = store.find(known, name, folder)
        if entry and entry.get("synopsis") and entry.get("fingerprint") == fingerprint:
            continue
        todo.append((name, folder, fingerprint,
                     chapter_text(entries, page_notes(os.path.join(folder, work_name)))))
    if todo:
        todo.sort(key=lambda item: chapter_order(item[0]))
        log(f"[Agent] Chapter synopses: writing {len(todo)} for the earlier chapters "
            f"({', '.join(name for name, *_ in todo)})")
        gate = asyncio.Semaphore(PARALLEL)
        totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "requests": 0}
        written: List[str] = []
        failed: List[str] = []

        async def one(name: str, folder: str, fingerprint: str, text: str) -> None:
            async with gate:
                if should_stop():
                    failed.append(f"{name} (stopped)")
                    return
                try:
                    synopsis, reply = await write_synopsis(client, settings, target_lang, name, text)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("agent: the synopsis of %s was not written", name, exc_info=True)
                    failed.append(f"{name} ({type(exc).__name__}: {str(exc)[:120]})")
                    return
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                totals[key] += int(getattr(reply, key, 0) or 0)
            totals["requests"] += 1
            if synopsis:
                store.put(name, folder, synopsis, fingerprint)
                written.append(name)
            else:
                failed.append(f"{name} (no synopsis in the reply)")

        await asyncio.gather(*(one(*item) for item in todo))
        written.sort(key=chapter_order)
        log(f"[Agent] Chapter synopses: {len(written)} written, {totals['prompt_tokens']} prompt tokens "
            f"({totals['cached_tokens']} cached), {totals['completion_tokens']} completion"
            + (f"; not written: {', '.join(failed)}" if failed else ""))
        info = {"written": written, "failed": failed, "usage": totals}
        known = store.load()
    return prompt_text(known, chapters), info
