"""The main agent's conversation: one chapter, page by page, in one conversation.

Context rules (design doc, "上下文與記憶管理"):

* the system prompt, tool definitions and memory snapshot come first and stay
  fixed until a compaction, so the prefix is cached;
* pages older than ``keep_images_last_n_pages`` keep their text but lose their images;
* at a page boundary, once a request reached ``compact_threshold_tokens``, the
  conversation restarts from the state files (memory + earlier results are
  reachable through tools);
* a page whose request passes ``max_context_tokens`` is stopped and flagged.

A failed page (API error, refusal, step limit) is flagged and the run moves on.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from .events import EventLog
from .page import PageDoc
from .settings import AgentSettings
from .store import AgentStore
from .tools import AgentSession, ToolRegistry

logger = logging.getLogger("manga_translator.agent")

try:
    from ..translators.common import VALID_LANGUAGES
except Exception:  # pragma: no cover - keeps the module importable in isolation
    VALID_LANGUAGES = {}


class AgentStopped(Exception):
    """The user stopped the run."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ChatReply:
    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class ChatClient(Protocol):
    async def complete(self, messages: List[dict], tools: List[dict], tool_choice: str) -> ChatReply: ...


DOUBLE_READ = {
    "all": "Zoom into every region for a second read before saving it.",
    "small_or_handwritten": ("Zoom for a second read whenever text is small, handwritten, stylized, on screentone, "
                             "or your first reading is uncertain; if the two reads disagree, zoom closer, and flag it "
                             "if they still disagree."),
    "none": "Zoom only when you cannot read the text from the page image.",
}

SYSTEM_PROMPT = """You are the translation agent inside a manga translation app. You work through one chapter page by page in a single conversation, so earlier pages are your context for later ones.

You are not a coding assistant here: you cannot run code, read files or browse. You act only by calling the tools provided; any other tool name fails.

Each page arrives as an image with numbered boxes plus a candidate list. A program found those boxes: it may have split one text block into several boxes or boxed something that is not text. The local OCR text in the list is a weak hint and is often wrong.

For every page:
1. Look at the numbered image. Use get_neighbors when a spread or a line of dialogue may continue across pages.
2. Fix the structure: merge_regions for one text block split over several boxes; skip_region for text that must stay as drawn (background signs, credits, sound effects woven into the art) and for boxes without text (not_text); set_reading_order if the order is wrong; link_regions for one sentence spread over several bubbles.
3. Read the text yourself from the image. {double_read}
4. Translate into {target}: natural, idiomatic manga dialogue that keeps each character's voice. Keep names and terms consistent with the glossary and character list; record new ones with glossary_upsert / character_upsert when they first appear.
5. Save every region with set_text: exact source transcription (furigana goes in ruby), the translation, category and confidence. Several set_text calls in one turn are fine.
6. When something is uncertain (unreadable text, a pun without a good equivalent, text the boxes missed), call flag_for_review with a short reason and keep going. Never stop to ask: no human is available until the run ends.
7. Finish with mark_page_done and your confidence for the page. It fails while any region has neither a translation nor a skip.

Write every reason, note and review flag in {reason_lang}; a human reads them.
Do not invent text that is not on the page. Keep translations short enough for their bubble; use measure_bubble when unsure. Update the story summary every few pages with story_summary_update.{extra}"""


def language_name(code: str) -> str:
    return VALID_LANGUAGES.get(code, code or "the target language")


def strip_images(message: dict, label: str) -> dict:
    content = message.get("content")
    if not isinstance(content, list):
        return message
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            parts.append({"type": "text", "text": f"[{label}: image omitted to save context]"})
        else:
            parts.append(part)
    return {**message, "content": parts}


class AgentRunner:
    def __init__(self, settings: AgentSettings, client: ChatClient, store: AgentStore, events: EventLog,
                 pages: List[str], target_lang: str,
                 should_stop: Callable[[], bool] = lambda: False):
        self.settings = settings
        self.client = client
        self.store = store
        self.events = events
        self.pages = pages
        self.target_lang = target_lang
        self.should_stop = should_stop
        self.memory = store.memory()
        self.session = AgentSession(settings, self.memory, pages, events)
        self.registry = ToolRegistry(self.session)
        self.preamble: List[dict] = []
        self.history: List[tuple] = []  # (page index, message)
        self.done_pages: List[str] = []
        self.totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "requests": 0}
        self._load_past()
        self._build_preamble()

    # --- conversation -------------------------------------------------------
    def system_prompt(self) -> str:
        target = language_name(self.target_lang)
        extra = f"\n\nAdditional instructions from the user:\n{self.settings.extra_instructions}" \
            if self.settings.extra_instructions.strip() else ""
        return SYSTEM_PROMPT.format(double_read=DOUBLE_READ.get(self.settings.double_read, DOUBLE_READ["small_or_handwritten"]),
                                    target=target, reason_lang=target, extra=extra)

    def _build_preamble(self, recap: str = "") -> None:
        intro = (f"This run covers {len(self.pages)} page(s): "
                 + ", ".join(os.path.basename(p) for p in self.pages) + ".\n\n" + self.memory.snapshot_text())
        if recap:
            intro += "\n\n" + recap
        self.preamble = [{"role": "system", "content": self.system_prompt()},
                         {"role": "user", "content": intro}]

    def _load_past(self) -> None:
        """Earlier results of this chapter (previous runs) for search_past_translations."""
        if not os.path.isdir(self.store.pages_dir):
            return
        current = {os.path.splitext(os.path.basename(p))[0] for p in self.pages}
        for name in sorted(os.listdir(self.store.pages_dir)):
            stem = name[: -len(".agent.json")] if name.endswith(".agent.json") else None
            if not stem or stem in current:
                continue
            side = self.store.load_sidecar(stem) or {}
            for region in side.get("regions", []):
                if region.get("translation"):
                    self.session.past.append({"page": stem, "id": region.get("id"),
                                              "source": region.get("ocr_agent") or region.get("ocr_local") or "",
                                              "translation": region["translation"], "category": region.get("category")})

    def request_messages(self, current_page: int) -> List[dict]:
        keep_from = current_page - self.settings.keep_images_last_n_pages + 1
        out = list(self.preamble)
        for page_index, message in self.history:
            if page_index < keep_from:
                message = strip_images(message, f"page {page_index + 1}")
            out.append(message)
        return out

    def compact(self, reason: str) -> None:
        recap = ""
        if self.done_pages:
            recap = ("Pages already finished in this run: " + ", ".join(self.done_pages)
                     + ". Their translations are saved; use search_past_translations to see how something was "
                       "translated before.")
            last = [p for p in self.session.past if p["page"] == self.done_pages[-1]]
            if last:
                recap += "\nLast finished page:\n" + "\n".join(
                    f"- #{p['id']} {p['source']} → {p['translation']}" for p in last)
        dropped = len(self.history)
        self.history = []
        self._build_preamble(recap)
        self.events.emit("compact", reason=reason, dropped_messages=dropped,
                         tokens_before=self.session.last_prompt_tokens)
        self.session.last_prompt_tokens = 0

    async def _complete(self, page_index: int) -> ChatReply:
        reply = await self.client.complete(self.request_messages(page_index), self.registry.schemas(), "required")
        self.session.last_prompt_tokens = reply.prompt_tokens
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            self.totals[key] += getattr(reply, key)
        self.totals["requests"] += 1
        self.events.emit("usage", page=self.session.page_name, prompt_tokens=reply.prompt_tokens,
                         completion_tokens=reply.completion_tokens, cached_tokens=reply.cached_tokens,
                         limit=self.settings.max_context_tokens)
        return reply

    def _page_message(self, index: int, doc: PageDoc) -> dict:
        from . import images
        regions = doc.active_regions()
        text = (f"Page {index + 1} of {len(self.pages)}: {os.path.basename(doc.image_path)}, "
                f"{len(regions)} candidate region(s).\n{self.session.candidate_lines()}\n"
                "Work through this page with the tools, then call mark_page_done.")
        image = images.draw_marks(self.session.page_image, self.session.marks())
        return {"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": images.data_url(image), "detail": self.settings.image_detail}},
        ]}

    # --- pages --------------------------------------------------------------
    async def run_page(self, index: int, doc: PageDoc) -> Dict[str, Any]:
        st = self.settings
        if self.session.last_prompt_tokens >= st.compact_threshold_tokens:
            self.compact("threshold")
        self.session.start_page(index, doc)
        page = doc.stem
        self.events.emit("page_start", page=page, index=index, total=len(self.pages),
                         regions=len(doc.active_regions()))
        self.history.append((index, self._page_message(index, doc)))
        done = False
        for step in range(st.max_steps_per_page):
            if self.should_stop():
                raise AgentStopped()
            self.session.page_steps = step + 1
            reply = await self._complete(index)
            if reply.prompt_tokens > st.max_context_tokens:
                doc.flag(None, "error", f"context limit reached ({reply.prompt_tokens} tokens)")
                self.events.emit("flag", page=page, region=None, flag="error", reason="context limit")
                break
            assistant = {"role": "assistant", "content": reply.content or None}
            if reply.tool_calls:
                assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                            "function": {"name": c.name, "arguments": c.arguments}}
                                           for c in reply.tool_calls]
            self.history.append((index, assistant))
            if not reply.tool_calls:
                self.history.append((index, {"role": "user", "content":
                                             "Continue with tool calls and finish the page with mark_page_done."}))
                continue
            for call in reply.tool_calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": call.arguments}
                self.events.emit("tool_call", page=page, tool=call.name, args=_compact_args(args),
                                 reason=args.get("reason", "") if isinstance(args, dict) else "")
                result = self.registry.execute(call.name, call.arguments)
                self.events.emit("tool_result", page=page, tool=call.name, ok=result.ok, summary=result.summary,
                                 images=len(result.images))
                content: Any = result.text
                if result.images:
                    content = [{"type": "text", "text": result.text}] + [
                        {"type": "image_url", "image_url": {"url": url, "detail": "high"}} for url in result.images]
                self.history.append((index, {"role": "tool", "tool_call_id": call.id, "content": content}))
                done = done or result.page_done
            if done:
                break
        if not done and doc.status != "done":
            doc.status = "incomplete"
            reason = f"stopped after {self.session.page_steps} steps; missing " + \
                     (", ".join(f"#{i}" for i in doc.missing()) or "nothing")
            doc.flag(None, "step_limit", reason)
            self.events.emit("flag", page=page, region=None, flag="step_limit", reason=reason)
        return self._finish_page(doc)

    def _finish_page(self, doc: PageDoc) -> Dict[str, Any]:
        doc.save()
        self.store.save_sidecar(doc.stem, doc.sidecar())
        self.session.finish_page()
        self.done_pages.append(doc.stem)
        outcome = {"page": doc.stem, "image": doc.image_path, "status": doc.status, "confidence": doc.confidence,
                   "flags": len(doc.flags), "missing": doc.missing()}
        self.events.emit("page_done", page=doc.stem, status=doc.status, confidence=doc.confidence,
                         flags=len(doc.flags), missing=doc.missing(), memory=self.memory.counts())
        return outcome

    async def run(self) -> List[Dict[str, Any]]:
        self.events.emit("run_start", pages=[os.path.basename(p) for p in self.pages],
                         model=getattr(self.client, "model_label", ""), target_lang=self.target_lang,
                         memory=self.memory.counts())
        outcomes: List[Dict[str, Any]] = []
        failures_in_a_row = 0
        try:
            for index, path in enumerate(self.pages):
                if self.should_stop():
                    raise AgentStopped()
                stem = os.path.splitext(os.path.basename(path))[0]
                try:
                    doc = PageDoc.load(path)
                except Exception as exc:
                    self.events.emit("error", page=stem, error=f"cannot open page data: {exc}")
                    outcomes.append({"page": stem, "image": path, "status": "error", "error": str(exc)})
                    continue
                try:
                    outcomes.append(await self.run_page(index, doc))
                    failures_in_a_row = 0
                except AgentStopped:
                    doc.status = "stopped"
                    self._finish_page(doc)
                    raise
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(f"agent page {stem} failed")
                    failures_in_a_row += 1
                    doc.status = "error"
                    doc.flag(None, "error", f"{type(exc).__name__}: {str(exc)[:300]}")
                    self.events.emit("error", page=stem, error=f"{type(exc).__name__}: {str(exc)[:300]}")
                    outcomes.append(self._finish_page(doc))
                    # The failed page's half-finished turn must not leak into the next page.
                    self.history = [(i, m) for i, m in self.history if i != index]
                    if failures_in_a_row >= 3:
                        self.events.emit("error", error="three pages failed in a row; stopping the run")
                        break
        finally:
            self.events.emit("run_end", pages=len(outcomes),
                             done=sum(1 for o in outcomes if o.get("status") == "done"),
                             flagged=sum(o.get("flags", 0) for o in outcomes), totals=self.totals,
                             memory=self.memory.counts())
        return outcomes


def _compact_args(args: Any, limit: int = 300) -> Any:
    """Arguments for the event log, with long strings shortened."""
    if isinstance(args, dict):
        return {k: _compact_args(v, limit) for k, v in args.items()}
    if isinstance(args, list):
        return [_compact_args(v, limit) for v in args[:50]]
    if isinstance(args, str) and len(args) > limit:
        return args[: limit - 1] + "…"
    return args
