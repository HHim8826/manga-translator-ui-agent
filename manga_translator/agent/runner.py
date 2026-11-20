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

Recall check (``recall_pass``): before the agent sees a page, a separate call
lists text that no box covers; each item is snapped to its strokes and added as
a region. It does not enter the main conversation, and the call for the next
page runs while the agent works on the current one.
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
from .snap import snap_text
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


RECALL_KINDS = ("dialogue", "narration", "handwritten", "sfx", "sign", "ui", "punctuation", "other")
RECALL_TOOL = {"type": "function", "function": {
    "name": "report_unboxed_text",
    "description": "Report every piece of text that no numbered box covers (empty list if there is none).",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {
        "items": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["box", "text", "kind", "direction"],
            "properties": {
                "box": {"type": "array", "items": {"type": "number"},
                        "description": "[x1, y1, x2, y2] in pixels of this image, tight around the text itself"},
                "text": {"type": "string", "description": "The text as written"},
                "kind": {"type": "string", "enum": list(RECALL_KINDS)},
                "direction": {"type": "string", "enum": ["v", "h"]}}}}}}}}

RECALL_PROMPT = """You check manga pages for text that an automatic text detector missed. Red numbered boxes on the page mark the text it found.

List every piece of text that is not inside a numbered box: speech and thoughts, narration, handwritten notes, small or faint text, text in tables, menus and status screens, signs, and sound effects. Give each separate text block its own item with a box in pixels of the image, tight around the letters themselves (not the whole bubble). Use kind "punctuation" for bubbles that hold only punctuation such as …, ? or !. Do not report text that is already inside a box, and do not report art that only looks like letters. Report an empty list when everything is boxed. Answer only by calling report_unboxed_text once."""


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
2. Fix the structure: merge_regions for one text block split over several boxes; skip_region for text that stays as drawn (background signs, credits) and for boxes without text (not_text); set_reading_order if the order is wrong; link_regions for one sentence spread over several bubbles.
   Sound effects drawn on the art (onomatopoeia such as impacts, footsteps, breathing outside a bubble) are never translated: skip_region with category sfx, even when they are large or easy to read. Text inside a speech bubble is dialogue and gets translated, even when it is a sound.
   Text without a box: add_region with a rough box in page pixels (the page size is in the page message), or with grid cells of a zoom (zoom also takes a box, to look at any area). The program snaps it to the strokes and shows what will be erased; undo if it is wrong. Then handle the new region like any other. Bubbles that hold only punctuation (…, ?, !) can stay as drawn without a box.
3. Read the text yourself from the image. {double_read}
4. Translate into {target}: natural, idiomatic manga dialogue that keeps each character's voice. Keep names and terms consistent with the glossary and character list; record new ones with glossary_upsert / character_upsert when they first appear.
5. Save every region with set_text: exact source transcription (furigana goes in ruby), the translation, category and confidence. Several set_text calls in one turn are fine.
6. When something is uncertain (unreadable text, a pun without a good equivalent, text that add_region cannot capture), call flag_for_review with a short reason and keep going. Never stop to ask: no human is available until the run ends.
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
        self.totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "requests": 0,
                       "recall_requests": 0}
        self._recall_tasks: Dict[int, asyncio.Future] = {}
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

    # --- recall check -------------------------------------------------------
    def _start_recall(self, index: int) -> None:
        if self.settings.recall_pass and 0 <= index < len(self.pages) and index not in self._recall_tasks:
            self._recall_tasks[index] = asyncio.ensure_future(self._recall_request(self.pages[index]))

    async def _recall_request(self, path: str) -> List[dict]:
        from . import images
        doc = PageDoc.load(path)
        page = images.open_page(path)
        marked = images.draw_marks(page, [(r.id, r.bbox, "pending") for r in doc.active_regions()])
        messages = [{"role": "system", "content": RECALL_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"This page is {page.width}x{page.height} pixels."},
                        {"type": "image_url", "image_url": {"url": images.data_url(marked),
                                                            "detail": self.settings.image_detail}}]}]
        reply = await self.client.complete(messages, [RECALL_TOOL], "required")
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            self.totals[key] += getattr(reply, key)
        self.totals["recall_requests"] += 1
        for call in reply.tool_calls:
            if call.name == RECALL_TOOL["function"]["name"]:
                items = json.loads(call.arguments or "{}").get("items", [])
                return items if isinstance(items, list) else []
        return []

    async def _recall(self, index: int, doc: PageDoc) -> List[int]:
        """Add the regions the recall check found; returns their ids."""
        self._start_recall(index)
        task = self._recall_tasks.pop(index)
        self._start_recall(index + 1)  # the next page's check runs while the agent works on this one
        try:
            items = await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"recall check failed for {doc.stem}: {exc}")
            self.events.emit("recall", page=doc.stem, added=[], error=f"{type(exc).__name__}: {str(exc)[:200]}")
            return []
        image = self.session.page_array
        added: List[int] = []
        ignored = {"punctuation": 0, "duplicate": 0, "no_strokes": 0, "invalid": 0}
        for item in items[:40]:
            if not isinstance(item, dict):
                ignored["invalid"] += 1
                continue
            box, kind = item.get("box"), str(item.get("kind") or "other")
            direction = "h" if str(item.get("direction") or "v").lower().startswith("h") else "v"
            if kind == "punctuation":
                ignored["punctuation"] += 1
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                ignored["invalid"] += 1
                continue
            if doc.overlap((x1, y1, x2, y2)) > 0.6:  # already boxed
                ignored["duplicate"] += 1
                continue
            snap = snap_text(image, (x1, y1, x2, y2), direction, exclude=doc.occupied(image.shape))
            if snap is None:
                ignored["no_strokes"] += 1
                continue
            if doc.overlap(snap.bbox) > 0.5:
                ignored["duplicate"] += 1
                continue
            text = str(item.get("text") or "").strip()
            added.append(doc.add_region(snap, direction, origin="recall", hint=text, kind=kind).id)
        doc.clear_history()  # undo is for the agent's own edits
        self.events.emit("recall", page=doc.stem, added=added, ignored=ignored, items=len(items))
        return added

    def _page_message(self, index: int, doc: PageDoc, recalled: Optional[List[int]] = None) -> dict:
        from . import images
        regions = doc.active_regions()
        page = self.session.page_image
        text = (f"Page {index + 1} of {len(self.pages)}: {os.path.basename(doc.image_path)} "
                f"({page.width}x{page.height} px), {len(regions)} candidate region(s).\n"
                f"{self.session.candidate_lines()}\n")
        if recalled:
            text += ("A recall check added " + ", ".join(f"#{i}" for i in recalled)
                     + " for text the detector missed; its reading is only a hint. Handle them like the others "
                       "(skip_region not_text if one is not really text).\n")
        text += "Work through this page with the tools, then call mark_page_done."
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
        recalled = await self._recall(index, doc) if st.recall_pass else []
        self.history.append((index, self._page_message(index, doc, recalled)))
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
            pending = list(self._recall_tasks.values())
            self._recall_tasks.clear()
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
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
