"""Optional proofreading pass (setting ``proofread``): after the translation agent has finished a
chapter, a separate conversation reads every translated region of the chapter at once and fixes
what is really wrong (meaning, names and terms, consistency across pages, wrong characters,
length). Each change is written to the page's JSON right away, with the old text and the reason
kept in the sidecar's ``proofread`` record.

It works on the saved JSON regions, so region numbers here are the JSON order (1, 2, ...), which
is also what the editor shows. It only changes translations: never the mask or the inpainted
image, so it can run while the inpaint review is still busy with the last pages.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import checks, images
from .events import EventLog
from .page import BREAK, Region, needs_break, text_length, with_breaks
from .settings import AgentSettings
from .store import AgentMemory, AgentStore, normalize, write_json_atomic

logger = logging.getLogger("manga_translator.agent")

PROOF_PROMPT = """You proofread a manga chapter that another agent translated into {target}. You get every translated region of the chapter at once: page, region number, category, about how many characters fit in the box, the source text and the translation.

Check each region for:
- meaning: mistranslations, omissions, the wrong speaker or addressee, the wrong level of politeness, jokes and wordplay that were lost;
- names and terms: the glossary and character list below, and the same name or term translated the same way on every page;
- language: natural {target} manga dialogue that fits the character; typos; wrong characters for {target} (for example simplified or Japanese forms such as 値 for 值 in Traditional Chinese); punctuation that matches the rest of the chapter;
- length: a translation much longer than its box holds.

Change only what is wrong or clearly better, and keep the translator's choices and style otherwise: do not rewrite lines that are fine. [BR] in a translation is a line break the renderer follows: keep the breaks when you revise a line, at natural boundaries of the new text. Use view_page when you need the picture (who is speaking, what is shown). Put all your revise_text calls in as few responses as you can, use flag_for_review for what you cannot settle, and end with finish_proofread. Write every reason and note in {reason_lang}; a human reads them.{extra}"""
PROOF_BREAKS = """

Line breaks: the user's render setting breaks lines only at [BR] (AI line breaks). The listing gives each region's number of source lines. A revised translation of a source with two or more lines needs at least one [BR], about one line per source line where natural boundaries allow it; a one-line source gets none. Never break inside a word or name, and never write a newline instead of [BR]."""


def _obj(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


PAGE = {"type": "string", "description": "Page name as listed, like 005"}
RID = {"type": "integer", "description": "Region number on that page"}
REASON = {"type": "string", "description": "Short reason a human can read"}

TOOLS = [
    {"type": "function", "function": {
        "name": "view_page", "description": "The page image with its region numbers.",
        "parameters": _obj({"page": PAGE}, ["page"])}},
    {"type": "function", "function": {
        "name": "revise_text", "description": "Replace one region's translation.",
        "parameters": _obj({"page": PAGE, "id": RID, "translation": {"type": "string"}, "reason": REASON},
                           ["page", "id", "translation", "reason"])}},
    {"type": "function", "function": {
        "name": "flag_for_review", "description": "Put a region (or a whole page, without id) on the human review list.",
        "parameters": _obj({"page": PAGE, "id": RID, "reason": REASON}, ["page", "reason"])}},
    {"type": "function", "function": {
        "name": "finish_proofread", "description": "End the proofreading, with an optional note for the human.",
        "parameters": _obj({"note": {"type": "string"}}, [])}},
]


class ProofError(ValueError):
    """Bad tool arguments; the message goes back to the model."""


class ProofPage:
    def __init__(self, image_path: str):
        from ..utils.path_manager import find_json_path

        self.image_path = image_path
        self.stem = os.path.splitext(os.path.basename(image_path))[0]
        self.json_path = find_json_path(image_path)
        if not self.json_path or not os.path.exists(self.json_path):
            raise FileNotFoundError(f"no translation JSON for {image_path}")
        with open(self.json_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.key = next(iter(raw))
        data = raw[self.key]
        regions = data.get("regions", []) if isinstance(data, dict) else data
        self.regions = {i: Region(id=i, data=r) for i, r in enumerate(regions, start=1)}
        self.categories: Dict[int, str] = {}
        """From the agent's sidecar, matched by box center."""

    def match_sidecar(self, side: Optional[dict]) -> None:
        by_center = {tuple((e.get("match") or {}).get("center") or ()): e for e in (side or {}).get("regions", [])}
        for rid, region in self.regions.items():
            entry = by_center.get(tuple(region.center))
            if entry and entry.get("category"):
                self.categories[rid] = str(entry["category"])

    @staticmethod
    def source(region: Region) -> str:
        return str(region.data.get("text") or "".join(region.data.get("texts") or []) or "")

    @staticmethod
    def translation(region: Region) -> str:
        return str(region.data.get("translation") or "")

    def listing(self, line_counts: bool = False) -> List[str]:
        """One line per translated region; the translation keeps its [BR] line breaks."""
        lines = []
        for rid, region in self.regions.items():
            translation = self.translation(region)
            if not translation:
                continue
            category = self.categories.get(rid, "")
            count = f", {region.line_count} line(s)" if line_counts else ""
            lines.append(f"#{rid} [{category + ', ' if category else ''}~{checks._capacity(region)} chars{count}] "
                         f"「{_one_line(self.source(region))}」 → 「{with_breaks(translation)}」")
        return lines

    def write(self, rid: int, translation: str) -> None:
        """Re-read the JSON and change only this region's translation (the inpaint review may have
        written the mask in the meantime)."""
        with open(self.json_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        data = raw[self.key]
        regions = data.get("regions", []) if isinstance(data, dict) else data
        entry = regions[rid - 1]
        entry["translation"] = translation
        entry["translation_raw"] = translation
        entry.pop("translation_rich", None)
        write_json_atomic(self.json_path, raw)
        self.regions[rid].data.update(translation=translation, translation_raw=translation)


def _one_line(text: str) -> str:
    return " ".join(str(text).replace(BREAK, " ").split())


class ProofSession:
    def __init__(self, pages: Dict[str, ProofPage], work_terms, memory: AgentMemory, target_lang: str,
                 line_breaks: bool = False):
        self.pages = pages
        self.work_terms = work_terms
        self.memory = memory
        self.target_lang = target_lang
        self.line_breaks = line_breaks
        """AI line breaks: a revision of a multi-line source must keep [BR] line breaks."""
        self.revisions: List[Dict[str, Any]] = []
        self.flags: List[Dict[str, Any]] = []
        self.note = ""
        self.finished = False

    def page(self, name: Any) -> ProofPage:
        key = os.path.splitext(str(name or "").strip())[0]
        if key not in self.pages:
            raise ProofError(f"there is no page {name!r}; pages: {', '.join(self.pages)}")
        return self.pages[key]

    def region(self, page: ProofPage, rid: Any) -> Region:
        try:
            rid = int(str(rid).lstrip("#"))
        except ValueError:
            raise ProofError(f"region number must be a number, got {rid!r}")
        region = page.regions.get(rid)
        if region is None or not page.translation(region):
            raise ProofError(f"page {page.stem} has no translated region #{rid}")
        return region


def _execute(s: ProofSession, name: str, args: Dict[str, Any]):
    """(text for the model, summary for the timeline, images)"""
    if name == "view_page":
        page = s.page(args.get("page"))
        image = images.open_page(page.image_path)
        marked = images.draw_marks(image, [(rid, r.bbox, "done") for rid, r in page.regions.items()
                                           if page.translation(r)])
        return f"Page {page.stem} with its region numbers.", page.stem, [images.data_url(marked, max_side=1600)]
    if name == "revise_text":
        page = s.page(args.get("page"))
        region = s.region(page, args.get("id"))
        new = with_breaks(args.get("translation"))
        reason = str(args.get("reason") or "").strip()
        if not new.strip():
            raise ProofError("translation is empty")
        if not reason:
            raise ProofError("reason is required")
        if s.line_breaks and needs_break(region.line_count, new):
            raise ProofError(f"{page.stem} #{region.id}: the source has {region.line_count} lines, so the new "
                             f"translation needs [BR] line breaks at natural boundaries; without them it is "
                             f"typeset as one line. Send it again with [BR].")
        old = page.translation(region)
        if normalize(new) == normalize(old):
            return f"{page.stem} #{region.id}: unchanged (same text).", "unchanged", []
        page.write(region.id, new)
        s.revisions.append({"page": page.stem, "id": region.id, "source": page.source(region), "old": old,
                            "new": new, "reason": reason})
        warnings = [checks.agent_text(i) for i in
                    checks.term_issues(region.id, page.source(region), new, s.work_terms, s.memory)]
        if s.target_lang != "JPN" and checks.KANA.search(new):
            warnings.append(f"#{region.id}: kana left in the translation")
        capacity = checks._capacity(region)
        if text_length(new) > checks.OVERFLOW_RATIO * capacity:
            warnings.append(f"#{region.id}: longer than the box holds (about {capacity} characters)")
        text = f"{page.stem} #{region.id} revised."
        if warnings:
            text += " Check: " + "; ".join(warnings)
        return text, f"{page.stem} #{region.id}: {_short(old)} → {_short(new)}", []
    if name == "flag_for_review":
        page = s.page(args.get("page"))
        rid = args.get("id")
        if rid is not None:
            rid = s.region(page, rid).id
        reason = str(args.get("reason") or "").strip()
        if not reason:
            raise ProofError("reason is required")
        s.flags.append({"page": page.stem, "region": rid, "type": "uncertain_translation", "reason": reason})
        return "Added to the review list.", f"{page.stem}" + (f" #{rid}" if rid else ""), []
    if name == "finish_proofread":
        s.note = str(args.get("note") or "")
        s.finished = True
        return "Proofreading finished.", f"{len(s.revisions)} revised", []
    raise ProofError(f"there is no tool named {name!r}")


def _short(text: str, limit: int = 24) -> str:
    text = _one_line(str(text).replace(BREAK, " / "))
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Proofreader:
    def __init__(self, settings: AgentSettings, client, store: AgentStore, events: EventLog, target_lang: str,
                 memory: AgentMemory, work_prompt=None, should_stop: Callable[[], bool] = lambda: False,
                 line_breaks: bool = False):
        self.settings = settings
        self.client = client
        self.store = store
        self.events = events
        self.target_lang = target_lang
        self.memory = memory
        self.work_prompt = work_prompt
        self.should_stop = should_stop
        self.line_breaks = line_breaks
        """AI line breaks (render setting disable_auto_wrap): revisions keep [BR] line breaks."""
        self.totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "requests": 0}

    def system_prompt(self) -> str:
        from .runner import language_name
        target = language_name(self.target_lang)
        return PROOF_PROMPT.format(target=target, reason_lang=target,
                                   extra=PROOF_BREAKS if self.line_breaks else "")

    def first_message(self, pages: Dict[str, ProofPage]) -> str:
        parts = []
        if self.work_prompt is not None:
            parts.append(f"Notes for this work from the user's prompt file {self.work_prompt.name}; its glossary "
                         f"is authoritative:\n{self.work_prompt.text}")
        parts.append("The translator's memory for this chapter:\n" + self.memory.snapshot_text())
        listed = []
        count = 0
        for stem, page in pages.items():
            lines = page.listing(self.line_breaks)
            count += len(lines)
            listed.append(f"Page {stem} ({len(lines)} region(s)):\n" + ("\n".join(lines) if lines else "(no text)"))
        parts.append(f"The chapter: {len(pages)} page(s), {count} translated region(s).\n\n" + "\n\n".join(listed))
        return "\n\n".join(parts)

    async def run(self, image_paths: Sequence[str]) -> Dict[str, Any]:
        from .runner import AgentStopped, _compact_args

        pages: Dict[str, ProofPage] = {}
        for path in image_paths:
            try:
                page = ProofPage(path)
            except Exception as exc:
                logger.warning(f"proofread: cannot open {path}: {exc}")
                continue
            page.match_sidecar(self.store.load_sidecar(page.stem))
            pages[page.stem] = page
        regions = sum(len(p.listing()) for p in pages.values())
        if not regions:
            return {"status": "skipped", "revised": 0, "flagged": 0}
        session = ProofSession(pages, getattr(self.work_prompt, "terms", []) if self.work_prompt else [],
                               self.memory, self.target_lang, self.line_breaks)
        self.events.emit("proofread_start", agent="proofread", pages=len(pages), regions=regions)
        messages: List[dict] = [{"role": "system", "content": self.system_prompt()},
                                {"role": "user", "content": self.first_message(pages)}]
        steps = 0
        for steps in range(1, self.settings.proofread_steps + 1):
            if self.should_stop():
                raise AgentStopped()
            reply = await self.client.complete(messages, TOOLS, "required", effort=self.settings.reasoning_effort)
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                self.totals[key] += getattr(reply, key)
            self.totals["requests"] += 1
            assistant: Dict[str, Any] = {"role": "assistant", "content": reply.content or None}
            if reply.tool_calls:
                assistant["tool_calls"] = [{"id": c.id, "type": "function",
                                            "function": {"name": c.name, "arguments": c.arguments}}
                                           for c in reply.tool_calls]
            messages.append(assistant)
            if not reply.tool_calls:
                messages.append({"role": "user", "content": "Continue with tool calls; end with finish_proofread."})
                continue
            for call in reply.tool_calls:
                try:
                    args = json.loads(call.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ProofError("arguments must be a JSON object")
                except (json.JSONDecodeError, ProofError) as exc:
                    args, error = {}, str(exc)
                else:
                    error = ""
                page_name = os.path.splitext(str(args.get("page") or ""))[0] or None
                self.events.emit("tool_call", page=page_name, agent="proofread", tool=call.name,
                                 args=_compact_args(args), reason=str(args.get("reason") or ""))
                pictures: List[str] = []
                ok = not error
                if ok:
                    try:
                        text, summary, pictures = _execute(session, call.name, args)
                    except (ProofError, KeyError, TypeError, ValueError) as exc:
                        ok, text, summary = False, f"Error: {exc}", str(exc)[:80]
                else:
                    text, summary = f"Error: {error}", error[:80]
                if ok and call.name == "flag_for_review" and session.flags:
                    flag = session.flags[-1]
                    self.events.emit("flag", page=flag["page"], agent="proofread", region=flag["region"],
                                     flag=flag["type"], reason=flag["reason"])
                self.events.emit("tool_result", page=page_name, agent="proofread", tool=call.name, ok=ok,
                                 summary=summary, images=len(pictures))
                content: Any = text
                if pictures:
                    content = [{"type": "text", "text": text}] + [
                        {"type": "image_url", "image_url": {"url": url, "detail": self.settings.image_detail}}
                        for url in pictures]
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
            if session.finished:
                break
        self._save(session, steps)
        by_page = {}
        for item in session.revisions:
            by_page[item["page"]] = by_page.get(item["page"], 0) + 1
        self.events.emit("proofread_done", agent="proofread", revised=len(session.revisions),
                         flagged=len(session.flags), pages=by_page, finished=session.finished, steps=steps,
                         note=session.note, usage=dict(self.totals))
        return {"status": "done" if session.finished else "incomplete", "revised": len(session.revisions),
                "flagged": len(session.flags), "steps": steps}

    def _save(self, session: ProofSession, steps: int) -> None:
        """Sidecar per page: the revisions (old and new text, reason) and the review flags."""
        for stem, page in session.pages.items():
            revisions = [r for r in session.revisions if r["page"] == stem]
            flags = [{k: v for k, v in f.items() if k != "page"} for f in session.flags if f["page"] == stem]
            if not revisions and not flags:
                continue
            side = self.store.load_sidecar(stem) or {}
            side.pop("version", None)
            centers = {rid: r.center for rid, r in page.regions.items()}
            for revision in revisions:
                for entry in side.get("regions", []):  # the agent's region with the same box
                    if (entry.get("match") or {}).get("center") == centers[revision["id"]] and \
                            entry.get("translation") == revision["old"]:
                        entry["translation"] = revision["new"]
                        entry.setdefault("reasons", []).append(f"proofread: {revision['reason']}")
                        break
            side["review_flags"] = list(side.get("review_flags") or []) + flags
            side["proofread"] = {"revisions": [{k: v for k, v in r.items() if k != "page"} for r in revisions],
                                 "note": session.note, "finished": session.finished, "steps": steps}
            self.store.save_sidecar(stem, side)
