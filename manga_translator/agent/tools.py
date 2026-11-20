"""Tools the agent calls (design doc, "工具規格"), M2 subset.

Reading: get_candidates, zoom, get_neighbors, local_ocr, measure_bubble
Structure: merge_regions, skip_region, set_reading_order, link_regions
Text: set_text, set_line_breaks
Memory: glossary_search, glossary_upsert, character_get, character_upsert,
        story_summary_get, story_summary_update, search_past_translations
Flow: flag_for_review, mark_page_done, budget_status, undo

Rules shared by every tool: regions are named by id, never by coordinates;
every write takes a ``reason``; bad arguments return an error text the model
can act on instead of raising.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import images
from .page import CATEGORIES, FLAG_TYPES, PageDoc, RegionError
from .settings import AgentSettings
from .store import AgentMemory, normalize

logger = logging.getLogger("manga_translator.agent")

SKIP_CATEGORIES = ("sign", "sfx", "credit", "not_text", "duplicate", "other")


@dataclass
class ToolResult:
    text: str
    images: List[str] = field(default_factory=list)
    summary: str = ""
    ok: bool = True
    page_done: bool = False


class AgentSession:
    """Everything the tools act on during one run."""

    def __init__(self, settings: AgentSettings, memory: AgentMemory, pages: List[str], events=None):
        self.settings = settings
        self.memory = memory
        self.pages = pages
        self.events = events
        self.page_index = -1
        self.doc: Optional[PageDoc] = None
        self.page_image = None
        self.past: List[Dict[str, Any]] = []
        self.last_prompt_tokens = 0
        self.page_steps = 0

    @property
    def page_name(self) -> str:
        return self.doc.stem if self.doc else ""

    def start_page(self, index: int, doc: PageDoc) -> None:
        self.page_index = index
        self.doc = doc
        self.page_image = images.open_page(doc.image_path)
        self.page_steps = 0

    def finish_page(self) -> None:
        if self.doc is None:
            return
        for region in self.doc.active_regions():
            if region.translation:
                self.past.append({"page": self.doc.stem, "id": region.id, "source": region.source_text or region.ocr_local,
                                  "translation": region.translation, "category": region.category})

    def marks(self):
        out = []
        for region in self.doc.active_regions():
            state = "skipped" if region.skipped else ("done" if region.translation else "pending")
            out.append((region.id, region.bbox, state))
        return out

    def candidate_lines(self) -> str:
        lines = []
        for region in self.doc.active_regions():
            x1, y1, x2, y2 = region.bbox
            local = region.ocr_local.replace("\n", " / ") or "(empty)"
            lines.append(f"#{region.id}: {region.direction}, {x2 - x1}x{y2 - y1}px, font≈{region.font_size}px, "
                         f"local OCR (weak hint): 「{local}」, {region.status()}")
        return "\n".join(lines) if lines else "(no candidate regions on this page)"


Handler = Callable[[AgentSession, Dict[str, Any]], ToolResult]


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Handler
    writes: bool = False


def _obj(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


ID = {"type": "integer", "description": "Region number as drawn on the page image"}
IDS = {"type": "array", "items": {"type": "integer"}, "description": "Region numbers"}
REASON = {"type": "string", "description": "One short sentence: why. Shown to the human on the Agents page"}


def _short(text: str, limit: int = 40) -> str:
    text = (text or "").replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- reading ------------------------------------------------------------------
def get_candidates(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    image = images.draw_marks(s.page_image, s.marks())
    return ToolResult(
        text=f"Page {s.page_name}: red = pending, green = translated, gray = skipped.\n{s.candidate_lines()}",
        images=[images.data_url(image)], summary=f"{len(s.doc.active_regions())} regions")


def zoom(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc.get(a["id"])
    others = [poly.tolist() for r in s.doc.active_regions() if r.id != region.id for poly in r.polygons]
    image, info = images.zoom(s.page_image, region.bbox, others, scale=float(a.get("scale") or 3),
                              mask_others=a.get("mask_others", True) is not False)
    cell = info.get("grid", {}).get("cell_page_px")
    return ToolResult(
        text=(f"Zoom of #{region.id} at {info['scale']}x; other regions' text is grayed out. "
              f"Grid A1-H8, each cell ≈ {cell[0]}x{cell[1]} page px." if cell else f"Zoom of #{region.id}."),
        images=[images.data_url(image)], summary=f"#{region.id} ×{info['scale']}")


def get_neighbors(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    span = max(1, min(2, int(a.get("range") or 1)))
    parts, labels = [], []
    for offset in range(-span, span + 1):
        index = s.page_index + offset
        if offset == 0 or not 0 <= index < len(s.pages):
            continue
        parts.append(images.thumbnail(s.pages[index], 640))
        labels.append(f"{'previous' if offset < 0 else 'next'} page ({offset:+d})")
    if not parts:
        return ToolResult(text="This page has no neighbors in this run.", summary="none")
    sheet = images.side_by_side(parts, labels)
    return ToolResult(text="Thumbnails: " + ", ".join(labels), images=[images.data_url(sheet, max_side=1600)],
                      summary=", ".join(labels))


def local_ocr(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc.get(a["id"])
    prob = region.data.get("prob")
    return ToolResult(text=json.dumps({"id": region.id, "local_ocr": region.ocr_local, "detector_prob": prob},
                                      ensure_ascii=False),
                      summary=f"#{region.id}: {_short(region.ocr_local)}")


def measure_bubble(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc.get(a["id"])
    x1, y1, x2, y2 = region.bbox
    w, h = x2 - x1, y2 - y1
    fs = max(8, region.font_size or int(min(w, h) * 0.8))
    if region.direction.startswith("v"):
        cols, per = max(1, int(w // (fs * 1.1))), max(1, int(h // fs))
        fit = f"about {cols} column(s) × {per} characters at the original size"
    else:
        rows, per = max(1, int(h // (fs * 1.2))), max(1, int(w // fs))
        fit = f"about {rows} line(s) × {per} characters at the original size"
    return ToolResult(
        text=(f"#{region.id}: text box {w}x{h}px, {region.direction}, original font ≈{fs}px, {fit}. "
              "The renderer can grow into the surrounding bubble and shrink the font, so this is a guide, not a limit."),
        summary=f"#{region.id} {w}x{h}")


# --- structure ----------------------------------------------------------------
def merge_regions(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    kept = s.doc.merge(a["ids"], a.get("reason", ""))
    return ToolResult(text=f"Merged into #{kept.id}. Set its text again with set_text.",
                      summary="+".join(f"#{i}" for i in a["ids"]) + f" → #{kept.id}")


def skip_region(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    category = a.get("category") or "other"
    if category not in SKIP_CATEGORIES:
        raise RegionError(f"category must be one of {', '.join(SKIP_CATEGORIES)}")
    region = s.doc.skip(a["id"], category, a.get("reason", ""))
    return ToolResult(text=f"#{region.id} skipped ({category}): it stays as in the original and is not erased.",
                      summary=f"#{region.id} ({category})")


def set_reading_order(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    order = s.doc.set_order(a["ids"])
    return ToolResult(text="Reading order: " + ", ".join(f"#{i}" for i in order if s.doc.regions[i].active),
                      summary=" → ".join(f"#{i}" for i in a["ids"]))


def link_regions(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    group = s.doc.link(a["ids"], a.get("reason", ""))
    return ToolResult(text=f"Linked as sentence group {group}. Translate them so the parts read as one sentence.",
                      summary="+".join(f"#{i}" for i in a["ids"]))


# --- text ---------------------------------------------------------------------
def set_text(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc.set_text(a["id"], a.get("source_text", ""), a.get("translation", ""),
                            a.get("category") or "dialogue", a.get("confidence"), a.get("ruby", ""),
                            a.get("reason", ""))
    remaining = len(s.doc.missing())
    return ToolResult(text=f"#{region.id} saved. {remaining} region(s) on this page still need a result.",
                      summary=f"#{region.id}: {_short(region.source_text, 24)} → {_short(region.translation, 30)}")


def set_line_breaks(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    region = s.doc.set_line_breaks(a["id"], a.get("text", ""), a.get("reason", ""))
    return ToolResult(text=f"#{region.id} line breaks saved.", summary=f"#{region.id}: {_short(region.translation)}")


# --- memory -------------------------------------------------------------------
def glossary_search(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    hits = s.memory.glossary_search(a.get("query", ""))
    return ToolResult(text=json.dumps(hits, ensure_ascii=False) if hits else "No glossary entries match.",
                      summary=f"{_short(a.get('query', ''), 20)}: {len(hits)}")


def glossary_upsert(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    status = s.memory.glossary_upsert(a["source"], a["target"], a.get("note", ""), s.page_name)
    return ToolResult(text=f"Glossary entry {status}.", summary=f"{a['source']} → {a['target']} ({status})")


def character_get(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    found = s.memory.character_get(a.get("name", ""))
    return ToolResult(text=json.dumps(found, ensure_ascii=False) if found else "No matching character.",
                      summary=f"{_short(a.get('name', '') or 'all', 20)}: {len(found)}")


def character_upsert(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    fields = {k: a.get(k) for k in ("target_name", "reading", "gender", "speech_style", "notes")}
    status = s.memory.character_upsert(a["name"], fields, s.page_name)
    return ToolResult(text=f"Character {status}.", summary=f"{a['name']} ({status})")


def story_summary_get(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    return ToolResult(text=s.memory.story or "(no summary yet)", summary=f"{len(s.memory.story)} chars")


def story_summary_update(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    s.memory.story_update(a["summary"])
    return ToolResult(text="Story summary saved.", summary=f"{len(s.memory.story)} chars")


def search_past_translations(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    q = normalize(a.get("query", ""))
    if not q:
        raise RegionError("query is required")
    hits = [p for p in s.past if q in normalize(p["source"]) or q in normalize(p["translation"])][:20]
    return ToolResult(text=json.dumps(hits, ensure_ascii=False) if hits else "No earlier translation matches.",
                      summary=f"{_short(a.get('query', ''), 20)}: {len(hits)}")


# --- flow ---------------------------------------------------------------------
def flag_for_review(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    flag_type = a.get("type") or "other"
    if flag_type not in FLAG_TYPES:
        raise RegionError(f"type must be one of {', '.join(FLAG_TYPES)}")
    entry = s.doc.flag(a.get("id"), flag_type, a.get("reason", ""))
    if s.events:
        s.events.emit("flag", page=s.page_name, region=entry["region"], flag=flag_type, reason=entry["reason"])
    return ToolResult(text="Added to the review list. Keep going.",
                      summary=(f"#{entry['region']} " if entry["region"] else "") + flag_type)


def mark_page_done(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    missing = s.doc.missing()
    if missing:
        ids = ", ".join(f"#{i}" for i in missing)
        return ToolResult(text=(f"Not finished: {ids} have neither a translation nor a skip. "
                                "Call set_text or skip_region for each, then mark_page_done again."),
                          ok=False, summary=f"missing {ids}")
    confidence = max(0.0, min(1.0, float(a.get("confidence", 0))))
    s.doc.status = "done"
    s.doc.confidence = confidence
    s.doc.note = a.get("note", "")
    if confidence < s.settings.low_confidence_threshold:
        entry = s.doc.flag(None, "low_confidence", a.get("note") or f"page confidence {confidence:.2f}")
        if s.events:
            s.events.emit("flag", page=s.page_name, region=None, flag="low_confidence", reason=entry["reason"])
    return ToolResult(text="Page done.", summary=f"confidence {confidence:.2f}", page_done=True)


def budget_status(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    st = s.settings
    return ToolResult(text=json.dumps({
        "context_tokens": s.last_prompt_tokens, "compact_at": st.compact_threshold_tokens,
        "hard_limit": st.max_context_tokens, "steps_this_page": s.page_steps,
        "max_steps_per_page": st.max_steps_per_page, "page": f"{s.page_index + 1}/{len(s.pages)}"}),
        summary=f"{s.last_prompt_tokens} tokens")


def undo(s: AgentSession, a: Dict[str, Any]) -> ToolResult:
    label = s.doc.undo()
    return ToolResult(text=f"Undone: {label}.", summary=label)


TOOLS: List[Tool] = [
    Tool("get_candidates", "Numbered page image and the candidate list with current status. Call again to see progress.",
         _obj({}, []), get_candidates),
    Tool("zoom", "Magnified crop of one region with an A1-H8 grid; other regions' text is grayed out. "
         "Use it for small, handwritten, stylized or uncertain text (second read).",
         _obj({"id": ID, "scale": {"type": "number", "description": "1-6, default 3"},
               "mask_others": {"type": "boolean", "description": "Gray out other regions (default true)"}}, ["id"]),
         zoom),
    Tool("get_neighbors", "Thumbnails of the previous and next pages, for spreads and dialogue that continues.",
         _obj({"range": {"type": "integer", "description": "1 or 2"}}, []), get_neighbors),
    Tool("local_ocr", "The local OCR model's reading of a region. A weak hint only: it is often wrong on handwriting.",
         _obj({"id": ID}, ["id"]), local_ocr),
    Tool("measure_bubble", "Size of a region's text box and roughly how many characters fit.",
         _obj({"id": ID}, ["id"]), measure_bubble),
    Tool("merge_regions", "Merge regions that are one text block (for example one bubble split in two).",
         _obj({"ids": IDS, "reason": REASON}, ["ids", "reason"]), merge_regions, writes=True),
    Tool("skip_region", "Leave a region as it is in the original: not erased, not translated. Categories: "
         + ", ".join(SKIP_CATEGORIES) + ". Use not_text for detections that contain no text.",
         _obj({"id": ID, "category": {"type": "string", "enum": list(SKIP_CATEGORIES)}, "reason": REASON},
              ["id", "category", "reason"]), skip_region, writes=True),
    Tool("set_reading_order", "Reading order of the regions on this page (right-to-left manga order by default).",
         _obj({"ids": IDS}, ["ids"]), set_reading_order, writes=True),
    Tool("link_regions", "Mark regions as parts of one sentence that spans several bubbles.",
         _obj({"ids": IDS, "reason": REASON}, ["ids", "reason"]), link_regions, writes=True),
    Tool("set_text", "Save the exact source transcription and the translation of one region. "
         "Call it once per region; several calls in one turn are fine.",
         _obj({"id": ID,
               "source_text": {"type": "string", "description": "Exact transcription of the original text, without ruby"},
               "ruby": {"type": "string", "description": "Furigana / side readings, if any, as 'base(reading)' pairs"},
               "translation": {"type": "string", "description": "The translation that will be typeset"},
               "category": {"type": "string", "enum": list(CATEGORIES)},
               "confidence": {"type": "number", "description": "0-1: how sure you are of reading and translation"},
               "reason": REASON},
              ["id", "source_text", "translation", "category", "confidence"]), set_text, writes=True),
    Tool("set_line_breaks", "Put explicit line breaks (\\n) into a region's current translation.",
         _obj({"id": ID, "text": {"type": "string"}, "reason": REASON}, ["id", "text"]), set_line_breaks, writes=True),
    Tool("glossary_search", "Search the glossary (source or target).",
         _obj({"query": {"type": "string"}}, ["query"]), glossary_search),
    Tool("glossary_upsert", "Add or change a glossary term so later pages use the same translation.",
         _obj({"source": {"type": "string"}, "target": {"type": "string"}, "note": {"type": "string"},
               "reason": REASON}, ["source", "target", "reason"]), glossary_upsert, writes=True),
    Tool("character_get", "Look up characters (empty name lists all).",
         _obj({"name": {"type": "string"}}, []), character_get),
    Tool("character_upsert", "Add or update a character: translated name, reading, gender, how they speak.",
         _obj({"name": {"type": "string", "description": "Name as written in the source"},
               "target_name": {"type": "string"}, "reading": {"type": "string"}, "gender": {"type": "string"},
               "speech_style": {"type": "string"}, "notes": {"type": "string"}, "reason": REASON},
              ["name", "reason"]), character_upsert, writes=True),
    Tool("story_summary_get", "The running story summary.", _obj({}, []), story_summary_get),
    Tool("story_summary_update", "Replace the running story summary (keep it under ~300 words).",
         _obj({"summary": {"type": "string"}, "reason": REASON}, ["summary", "reason"]), story_summary_update,
         writes=True),
    Tool("search_past_translations", "Search earlier pages' source text and translations in this run.",
         _obj({"query": {"type": "string"}}, ["query"]), search_past_translations),
    Tool("flag_for_review", "Put something on the human review list and keep going. Types: " + ", ".join(FLAG_TYPES),
         _obj({"id": ID, "type": {"type": "string", "enum": list(FLAG_TYPES)}, "reason": REASON},
              ["type", "reason"]), flag_for_review, writes=True),
    Tool("mark_page_done", "Finish the page. Fails while any region has neither a translation nor a skip.",
         _obj({"confidence": {"type": "number", "description": "0-1 for the whole page"},
               "note": {"type": "string", "description": "Anything the human should know about this page"}},
              ["confidence"]), mark_page_done, writes=True),
    Tool("budget_status", "Context size and steps used on this page.", _obj({}, []), budget_status),
    Tool("undo", "Undo the last change on this page.", _obj({}, []), undo, writes=True),
]


class ToolRegistry:
    def __init__(self, session: AgentSession, tools: Optional[List[Tool]] = None):
        self.session = session
        self.tools = {t.name: t for t in (tools or TOOLS)}

    def schemas(self) -> List[Dict[str, Any]]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.parameters}} for t in self.tools.values()]

    def execute(self, name: str, raw_args: Any) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(text=f"Error: there is no tool named {name!r}. Available: {', '.join(self.tools)}.",
                              ok=False, summary="unknown tool")
        try:
            args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
        except json.JSONDecodeError as exc:
            return ToolResult(text=f"Error: arguments are not valid JSON ({exc}).", ok=False, summary="bad JSON")
        if not isinstance(args, dict):
            return ToolResult(text="Error: arguments must be a JSON object.", ok=False, summary="bad arguments")
        missing = [p for p in tool.parameters.get("required", []) if args.get(p) in (None, "")]
        if missing:
            return ToolResult(text=f"Error: missing required argument(s): {', '.join(missing)}.", ok=False,
                              summary="missing " + ", ".join(missing))
        if self.session.doc is None and name not in ("glossary_search", "character_get", "story_summary_get"):
            return ToolResult(text="Error: no page is open.", ok=False, summary="no page")
        try:
            return tool.handler(self.session, args)
        except (RegionError, ValueError, KeyError, TypeError) as exc:
            return ToolResult(text=f"Error: {exc}", ok=False, summary=str(exc)[:80])
        except Exception as exc:  # a tool bug must not end the run
            logger.exception(f"agent tool {name} failed")
            return ToolResult(text=f"Error: the tool failed internally ({type(exc).__name__}: {exc}).", ok=False,
                              summary=f"{type(exc).__name__}")
