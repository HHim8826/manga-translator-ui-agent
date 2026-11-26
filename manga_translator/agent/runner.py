"""The main agent's conversation: one chapter, page by page, in one conversation.

Context rules:

* the system prompt, tool definitions and memory snapshot come first and stay
  fixed until a compaction, so the prefix is cached;
* each page starts afresh from them, its message led by the last pages' translations
  (``RECAP_PAGES``); earlier pages are reachable through the memory tools. With
  ``whole_chapter_conversation`` the earlier pages stay instead, and:
* pages older than ``keep_images_last_n_pages`` keep their text but lose their images,
  n pages at a time so the cached prefix changes once every n pages, and not when only a
  few pages are left (``AgentRunner.drop_images``);
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
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from . import json_guard
from .events import EventLog
from .font_styles import styles_text
from .page import PageDoc
from .render_check import RenderCheck
from .settings import AgentSettings
from .snap import snap_text
from .store import AgentStore
from .store import page_stem
from .tools import TOOLS, AgentSession, ToolRegistry
from .work_prompt import WorkPrompt

logger = logging.getLogger("manga_translator.agent")

try:
    from ..translators.common import VALID_LANGUAGES
except Exception:  # pragma: no cover - keeps the module importable in isolation
    VALID_LANGUAGES = {}

RECAP_PAGES = 2
"""A page started afresh sees the translations of this many pages before it."""
CONTEXT_IMAGE_SIDE = 1024
"""Longest side of the earlier pages' images (``context_page_images``): enough to tell the characters
apart; their text is in the recap."""


def _natural(stem: str) -> list:
    """Sort key of page names: 2 before 10."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part) for part in re.split(r"(\d+)", stem or "") if part]
DROP_PAGES_LEFT = 4
"""Old page images are dropped only while at least this many pages (the current one included) are left
(``AgentRunner.drop_images``)."""


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
    async def complete(self, messages: List[dict], tools: List[dict], tool_choice: str,
                       effort: str = "", cache_key: str = "") -> ChatReply: ...


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

FLOW = {
    True: ("You work through one chapter page by page in a single conversation, so earlier pages are your context "
           "for later ones.", "Update the story summary every few pages with story_summary_update."),
    False: ("You work through one chapter page by page, and each page starts a new conversation: what earlier pages "
            "settled reaches you through the glossary, character list and story summary below, the translations of "
            "the last pages at the top of the page message, and search_past_translations.",
            "Update the story summary with story_summary_update every few pages and whenever something happens that "
            "later pages need to know: they see the summary, not this page. Nor do they see who is who: when someone "
            "speaks or is spoken to on this page and the character list lacks them, add them with character_upsert, "
            "even when the glossary or the work's names already have them. Give their gender (it decides gendered "
            "words such as the pronoun used to address them), how they speak and address others, and in notes what "
            "they look like, so a later page can tell who is talking to whom. Update them when a page shows more."),
}
"""The prompt's lines on the conversation, by ``whole_chapter_conversation``."""

SYSTEM_PROMPT = """You are the translation agent inside a manga translation app. {flow}

You are not a coding assistant here: you cannot run code, read files or browse. You act only by calling the tools provided; any other tool name fails.

Each page arrives as an image with numbered boxes plus a candidate list. A program found those boxes: it may have split one text block into several boxes or boxed something that is not text. The local OCR text in the list is a weak hint and is often wrong.

For every page:
1. Look at the numbered image. Use get_neighbors when a spread or a line of dialogue may continue across pages.
2. Fix the structure: merge_regions only for one paragraph cut into several boxes (lines that follow each other with no blank space between them). Separate paragraphs stay separate regions even inside one bubble or narration box, so the typeset page keeps them apart: translate each where it is, and use link_regions if one sentence runs across them. skip_region for text that stays as drawn (background signs, credits) and for boxes without text (not_text); set_reading_order if the order is wrong; link_regions for one sentence spread over several bubbles.
   One box can also hold two separate texts: two bubbles, or the two parts of one joined bubble, each with its own line; a small aside or inner thought written beside the main text in smaller or handwritten letters; a caption beside a title; a subtitle printed in a logo. The candidate list names the boxes whose lines stand apart. split_region them apart before saving: translate each part where it is (an aside keeps its own small lettering, so give it a style when it looks different from the text beside it), and skip_region a part that stays as drawn, such as lettering that belongs to a title logo. Never drop a part: an aside gets its own translation too.
   The two texts can also share one detected line: a meaning gloss under invented glyphs (「これ」, 「私」), a name in kana written on the line of the glyphs, a word the detector joined to the next one. split_region cuts inside a line too: zoom, then give a box around only the strokes of that part. Translate the part and skip_region the glyphs and their readings (other): the program keeps the skipped strokes on the page while it erases the translated ones, even inside one line. Do not flag a gloss as impossible to box before you tried this.
   Sound effects drawn on the art (onomatopoeia such as impacts, footsteps, breathing outside a bubble) are never translated: skip_region with category sfx, even when they are large or easy to read. Text inside a speech bubble is dialogue and gets translated, even when it is a sound.
   The work's title logo (usually on the first page of a chapter) stays as drawn, all of it: the title lettering with its furigana, the subtitle or English title printed with it, and the author's name. skip_region each of them (category other for the logo and its subtitle, credit for the author's name), also a box that holds only part of the logo, and even when the glossary has a translation of the title. The chapter number and chapter name printed beside the logo are translated as usual.
   Text without a box: add_region with a rough box in page pixels (the page size is in the page message), or with grid cells of a zoom (zoom also takes a box, to look at any area). The program snaps it to the strokes and shows what will be erased; undo if it is wrong. Then handle the new region like any other. When a detected box took in some of the art (its font size then reads too large), cut off part of its text, or read two lines as one, fit_region it to the text. A label written on a slant (along a road or a roof) gets a slanted box from add_region and fit_region, and its translation is drawn along it; an upright box around a slanted label reads a huge font, so fit_region it.
   Text written over detailed art, where erasing it would break the drawing and its translation in its place would not hide the damage (a map's labels on screentone between dashed lines, words across hatching, hair, a face or an object): note_region it instead of set_text. Its strokes stay as drawn and its translation goes beside it in brackets, smaller, on the spot the program finds; fit_region a slanted label first, so the note follows the slant. Look at the image the result shows, and give a box when the note covers something that matters. Only that text: most text on a busy page, a map too, is still translated in its place: a label in its own box, circle or frame, a big name with its own white outline on a flat tone, text in a bubble or a caption, on plain paper, a flat tone or a gradient (white-outlined text on screentone erases cleanly). note_region refuses text on flat ground. Sound effects stay skipped. When add_region catches a neighbouring label too, undo and add it again with exact: true. Bubbles that hold only punctuation (…, ?, !) can stay as drawn without a box.
3. Read the text yourself from the image. {double_read}
4. Translate into {target}: natural, idiomatic manga dialogue that keeps each character's voice. Keep names and terms consistent with the glossary and character list; record new ones with glossary_upsert / character_upsert when they first appear.
   {invented_script}
5. Save every region with set_text: exact source transcription (furigana goes in ruby), the translation, category and confidence. Save the whole page in one response: every set_text and skip_region call (plus set_reading_order and glossary or character updates) together.
6. When something is uncertain (unreadable text, a pun without a good equivalent, text that add_region and split_region cannot capture), call flag_for_review with a short reason and keep going. Never stop to ask: no human is available until the run ends.
7. Finish with mark_page_done and your confidence for the page; it can go in the same response as the saves. It fails while any region has neither a translation nor a skip.

Each response takes several seconds and resends the whole conversation, so call independent tools together in one response: all the zooms you need at once, then all the structure changes at once, then all the saves at once. Changes to different boxes are independent: send split_region, merge_regions, fit_region and add_region for every box that needs one in one response; each result shows its own image and gives the new ids. Wait for a result only when your next call needs something from it: the id a split_region or add_region gives, a second change to the same box, a zoom you have to see first. Saves never wait for each other. undo takes back only the last change: to take back an earlier change of such a response, undo back to it and send the later ones again.
Write every reason, note and review flag in {reason_lang}; a human reads them.
Do not invent text that is not on the page. Keep translations short enough for their bubble; use measure_bubble when unsure. {story}{extra}"""


INVENTED_SCRIPT = ("Invented script: a made-up language may be drawn as invented glyphs with a kana reading beside "
                   "them and its meaning in Japanese (in 「」 or under a bracket). The reading is only how the words "
                   "sound, not what they say: translate the meaning, and do not write the reading or a "
                   "transliteration of it into the translation. Keep the sound only where the scene turns on it "
                   "(a character repeats or mishears the word). With no meaning on the page, use the one from "
                   "earlier pages or the glossary; when it is unknown, the glyphs stay as drawn (skip_region) "
                   "and get a review flag.")
"""Shared by the translation agent and the proofreader, so a revision does not put the readings back."""


FONT_PROMPT = """

Fonts: every region is typeset in the user's font ({default}) unless it has a font style. The font sheet in the first message shows the other installed fonts that have every {target} character, numbered. A font marked "no vertical ..." lacks those marks of vertical text (they are drawn from a similar font): for vertical text, prefer a font that has them when two look alike. A font style is a kind of lettering that clearly differs from the normal dialogue around it: a narration box in a serif face, handwriting, cute or pop lettering, brush writing, a sign, a title or a screen. Define each style once with set_font_style (a short name in {target}, the listed font that looks most like that lettering, and what the lettering looks like), then give that style in set_text for every region with that lettering, all through the chapter. The styles listed in the first message come from earlier chapters of this work: keep using them for the same kinds of text. Normal dialogue has no style, whoever speaks; give dialogue a style only when its lettering clearly differs, and say how in reason. The program keeps a style's font in a region only if the text still fits the same place, and draws characters a font lacks from a similar font."""

LINE_BREAK_PROMPT = """

Line breaks: the user's render setting breaks lines only where a translation has [BR] (AI line breaks), so write the line breaks into every translation. The candidate list gives each region's number of source lines; that is the original_region_count of the rules below. Write the marker [BR] in set_text, never a newline.
The renderer keeps every line as you break it and does not shrink the text, so each line has to fit the box: the candidate list also says about how many characters fit per line and how many lines fit. Fitting the box comes before the number of source lines: when a line would be longer, break the text into more lines of about even length.
{rules}"""
LINE_BREAK_RULES = """Insert [BR] only at natural boundaries (after a sentence or clause, between phrases), never inside a word, name or number. A one-line source gets no [BR]; a source of two or more lines gets at least one, and about one line per source line when natural boundaries allow it. Never put [BR] at the start or the end, or two in a row."""
"""Used when dict/system_prompt_line_break.yaml cannot be read."""

REVISE_PROMPT = """

Checking again: the pages of this run were translated by an earlier run, under older rules than the ones above. The candidate list gives every region's current translation (green boxes); gray boxes are text the earlier run skipped. Go over each page against its image and fix what the earlier run got wrong or missed, with every tool:
- a box that holds two separate texts (two bubbles, the two parts of a joined bubble, an aside beside the main text, a subtitle in a logo, a gloss or a name on the line of invented glyphs): split_region it and translate each part in its place; the list names the boxes whose lines stand apart, and their one translation may cover both parts or drop one;
- text without a box: add_region (the recall check adds what it finds, as on a new page);
- boxes that cut one paragraph apart: merge_regions; text that should stay as drawn: skip_region; skipped text that is dialogue or narration: set_text it (its strokes are then erased); text over detailed art whose erasing broke the drawing: note_region it (its strokes come back);
- a wrong or stiff translation, invented script written as its reading instead of its meaning, a name that differs from the glossary: set_text again;
- missing line breaks (set_line_breaks) and lettering that needs a font style (set_text with the style).
A translation that is right stays: it needs no call. Regions marked LOCKED were changed by the user in the editor after the earlier run: every write tool refuses them, so leave them, and flag_for_review one if it looks wrong. Where the user deleted a box, the text stays without one: add_region refuses it. Finish every page with mark_page_done."""

BATCHED_SAVES = ("set_text", "skip_region", "note_region")
BATCH_HINT = (" {left} region(s) are left: save them all in your next response (several set_text / skip_region "
              "calls in one response), not one per response.")
STRUCTURE_EDITS = ("split_region", "merge_regions", "fit_region", "add_region")
STRUCTURE_HINT = (" If other boxes need split_region, merge_regions, fit_region or add_region too, send them all in "
                  "your next response, not one per response.")
"""Said after a response that held one structure change: such responses were a quarter to two fifths
of a page's prompt tokens, as each resends the page."""


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
                 should_stop: Callable[[], bool] = lambda: False,
                 on_page_done: Optional[Callable[[Dict[str, Any]], None]] = None,
                 work_prompt: Optional[WorkPrompt] = None, fonts=None, line_breaks: str = "",
                 font_styles: Optional[Dict[str, dict]] = None, run_info: Optional[Dict[str, Any]] = None,
                 preprocess_failed: Optional[Dict[str, str]] = None, revise: bool = False,
                 revise_fonts=None, default_font: str = "", render_backend=None):
        self.settings = settings
        self.revise = revise
        """The pages were finished by an earlier run: check them again (``revise.py``)."""
        self.revise_fonts = revise_fonts if revise_fonts is not None else fonts
        """``fonts.FontService`` for the thin-font replacement of a check (also when the agent does not
        choose fonts); ``default_font`` is the user's font when there is none."""
        self.default_font = default_font
        self.preprocess_failed = {os.path.normcase(os.path.abspath(k)): v for k, v in (preprocess_failed or {}).items()}
        """Image path -> why the preprocess did not write the page again: the agent leaves it alone
        (its JSON is the one from before the run)."""
        self.run_info = dict(run_info or {})
        """Extra ``run_start`` fields from the workflow: ``chapter`` (folder name) and ``batch`` (one id
        for all chapters of a run, so the AI Agent page keeps the earlier chapters)."""
        self.work_prompt = work_prompt
        """The work's prompt file (instructions + glossary), shared with the classic translators."""
        self.fonts = fonts if fonts is not None and getattr(fonts, "choices", None) else None
        """``fonts.FontService`` with a font sheet for set_font_style, or None."""
        backend = render_backend if render_backend is not None else getattr(self.revise_fonts, "backend", None)
        self.render_check = RenderCheck(backend) if hasattr(backend, "load_page") else None
        """The size the render draws each region at, before mark_page_done (``render_check.py``);
        None without a renderer (``render_backend``, else the font service's)."""
        self.line_breaks = line_breaks
        """The AI line break rules when the render setting disable_auto_wrap is on, else empty."""
        self.client = client
        self.store = store
        self.events = events
        self.pages = pages
        self.target_lang = target_lang
        self.should_stop = should_stop
        self.on_page_done = on_page_done
        """Called with each page's outcome once its JSON and sidecar are saved (the inpaint review starts there)."""
        self.memory = store.memory()
        if font_styles and self.fonts is not None:  # the work's styles from earlier chapters
            self.memory.font_styles_seed(font_styles)
        self._replace_thin_styles()
        self.session = AgentSession(settings, self.memory, pages, events, target_lang=target_lang)
        self.session.work_terms = list(work_prompt.terms) if work_prompt else []
        self.session.font_choices = [face.value for face in self.fonts.choices] if self.fonts else []
        self.session.line_breaks = bool(line_breaks)
        # font styles need the font sheet
        self.registry = ToolRegistry(self.session, None if self.fonts is not None else
                                     [t for t in TOOLS if t.name != "set_font_style"])
        self.preamble: List[dict] = []
        self.history: List[tuple] = []  # (page index, message)
        self.images_from = 0
        """Pages before this one have lost their images in the conversation (``drop_images``)."""
        self.done_pages: List[str] = []
        self.totals = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "requests": 0,
                       "recall_requests": 0}
        self._recall_tasks: Dict[int, asyncio.Future] = {}
        self._load_past()
        self._build_preamble()

    # --- conversation -------------------------------------------------------
    def system_prompt(self) -> str:
        target = language_name(self.target_lang)
        extra = ""
        if self.work_prompt is not None:
            extra += (f"\n\nNotes for this work from the user's prompt file {self.work_prompt.name}. Follow them, and "
                      f"use its glossary over your own choices and over the glossary tools:\n{self.work_prompt.text}")
        if self.line_breaks:
            extra += LINE_BREAK_PROMPT.format(rules=self.line_breaks.strip())
        if self.fonts is not None:
            extra += FONT_PROMPT.format(default=self.fonts.default_font, target=target)
        if self.revise:
            extra += REVISE_PROMPT
        if self.settings.extra_instructions.strip():
            extra += f"\n\nAdditional instructions from the user:\n{self.settings.extra_instructions}"
        flow, story = FLOW[bool(self.settings.whole_chapter_conversation)]
        return SYSTEM_PROMPT.format(flow=flow, story=story,
                                    double_read=DOUBLE_READ.get(self.settings.double_read, DOUBLE_READ["small_or_handwritten"]),
                                    target=target, reason_lang=target, extra=extra,
                                    invented_script=INVENTED_SCRIPT)

    def _build_preamble(self, recap: str = "") -> None:
        intro = (f"This run covers {len(self.pages)} page(s): "
                 + ", ".join(os.path.basename(p) for p in self.pages) + ".\n\n" + self.memory.snapshot_text())
        if recap:
            intro += "\n\n" + recap
        content: Any = intro
        if self.fonts is not None:
            styles = styles_text(self.memory.font_styles, self._font_number)
            content = [{"type": "text", "text": intro + "\n\nFont sheet (set_font_style font):\n" + self.fonts.sheet_text()
                        + "\n\nFont styles (set_text style):\n" + styles}]
            if self.fonts.sheet_url:
                content.append({"type": "image_url", "image_url": {"url": self.fonts.sheet_url,
                                                                   "detail": self.settings.image_detail}})
        self.preamble = [{"role": "system", "content": self.system_prompt()},
                         {"role": "user", "content": content}]

    def _replace_thin_styles(self) -> None:
        """Styles kept from chapters before thin fonts were left out move to the sheet font that looks
        most like their font; the work's styles file takes the new font at the end of the run."""
        stand_in = getattr(self.fonts, "stand_in", None)
        if stand_in is None:
            return
        for name, entry in list(self.memory.font_styles.items()):
            old = entry.get("font", "")
            new = stand_in(old)
            if new is not None and new != old:
                self.memory.font_style_font(name, new)
                logger.info(f"[Agent] Font style {name}: {old} is a thin font, now {new or 'the user font'}")

    def _font_number(self, font: str) -> str:
        """``3 (MingLiU)`` for a style's font on this run's font sheet, the font name otherwise."""
        for number, value in enumerate(self.session.font_choices, start=1):
            if value == font:
                return f"font {number} ({value})"
        return font

    def _load_past(self) -> None:
        """Earlier results of this chapter (previous runs) for search_past_translations."""
        if not os.path.isdir(self.store.pages_dir):
            return
        current = {page_stem(p) for p in self.pages}
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

    def image_pages_from(self, current_page: int) -> int:
        """First page whose images may stay in the conversation. Images are dropped n pages at a time, so
        the start of the conversation (and the API's prompt cache) changes once every n pages instead of on
        every page; between n and 2n - 1 pages keep their images."""
        n = self.settings.keep_images_last_n_pages
        return max(0, ((current_page + 1) // n - 1) * n)

    def drop_images(self, index: int) -> None:
        """At the start of page ``index``: drop the next block of old images when it pays. A drop changes
        the conversation from the first dropped image on, so that request finds only the start cached and
        pays for the rest in full (ch34 check, page 10 of 10: 58k of 73k tokens), while each later request
        saves only the dropped images at the cached price. That takes about three pages to make up for,
        so with fewer than ``DROP_PAGES_LEFT`` pages left the images stay, unless the conversation is
        already past half the compaction threshold."""
        due = self.image_pages_from(index)
        if due <= self.images_from:
            return
        left = len(self.pages) - index
        if left >= DROP_PAGES_LEFT or self.session.last_prompt_tokens >= self.settings.compact_threshold_tokens // 2:
            self.images_from = due

    def request_messages(self, current_page: int) -> List[dict]:
        out = list(self.preamble)
        for page_index, message in self.history:
            if page_index < self.images_from:
                message = strip_images(message, f"page {page_index + 1}")
            out.append(message)
        return out

    def fresh_page(self) -> None:
        """Start the next page without the earlier pages' messages: the preamble again, with the memory as
        the earlier pages left it, so it stays cached while nothing was added to the memory."""
        self.history = []
        self.images_from = 0
        self._build_preamble()
        self.session.last_prompt_tokens = 0

    def _earlier_pages(self, current: str) -> List[str]:
        """The translated pages before ``current``, this run's and an earlier run's (a chapter is often run
        in parts), oldest first."""
        return sorted({p["page"] for p in self.session.past if _natural(p["page"]) < _natural(current)},
                      key=_natural)

    def _page_image(self, name: str, folder: str) -> Optional[str]:
        """The image of the page called ``name``: one of this run's, else one in ``folder`` (a page an
        earlier run translated)."""
        for path in self.pages:
            if page_stem(path) == name:
                return path
        from ..image_formats import SUPPORTED_IMAGE_EXTENSIONS
        try:
            files = sorted(os.listdir(folder))
        except OSError:
            return None
        for file in files:
            path = os.path.join(folder, file)
            if os.path.splitext(file)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS and page_stem(path) == name:
                return path
        return None

    def context_images(self, doc: PageDoc) -> Tuple[str, List[str]]:
        """The text and the image urls of the ``context_page_images`` translated pages before ``doc``;
        ("", []) when the setting is 0, the chapter is one conversation (its pages are in it already) or
        no earlier page has an image."""
        from . import images
        count = self.settings.context_page_images
        if not count or self.settings.whole_chapter_conversation:
            return "", []
        names, urls = [], []
        for name in self._earlier_pages(doc.stem)[-count:]:
            path = self._page_image(name, os.path.dirname(doc.image_path))
            if path is None:
                continue
            try:
                urls.append(images.data_url(images.thumbnail(path, CONTEXT_IMAGE_SIDE)))
            except Exception:  # an unreadable page: the others still help
                logger.warning("agent: the context image of page %s could not be read", name, exc_info=True)
                continue
            names.append(name)
        if not names:
            return "", []
        text = (f"Earlier page{'s' if len(names) > 1 else ''} for context only, oldest first: {', '.join(names)}. "
                "They are already translated: look at them to see who is who, who speaks to whom and what "
                "happened. Do not translate them; their numbered boxes and ids are not this page's.")
        return text, urls

    def recap(self, current: str) -> str:
        """The translated pages before ``current`` (this run's and an earlier run's: a chapter is often run
        in parts), and what the last ``RECAP_PAGES`` of them say."""
        earlier = self._earlier_pages(current)
        if not earlier:
            return ""
        lines = ["Pages before this one, already translated: " + ", ".join(earlier) + ". Their translations "
                 "are saved; use search_past_translations to see how something was translated before."]
        for stem in earlier[-RECAP_PAGES:]:
            entries = [p for p in self.session.past if p["page"] == stem]
            if entries:
                lines.append(f"Page {stem} as translated:")
                lines += [f"- #{p['id']} {p['source'] or ''} → {p['translation']}" for p in entries]
        return "\n".join(lines) + "\n\n"

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
        reply = await self.client.complete(self.request_messages(page_index), self.registry.schemas(), "required",
                                           effort=self.settings.reasoning_effort, cache_key="translator")
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

    def load_doc(self, path: str) -> PageDoc:
        """The page for the agent: the preprocess JSON, or on a check the page as it was left."""
        if not self.revise:
            return PageDoc.load(path)
        from .revise import load_page
        return load_page(path, self.store, lock=self.settings.revise_lock_user_edits, fonts=self.revise_fonts,
                         default_font=self.default_font)

    async def _recall_request(self, path: str) -> List[dict]:
        from . import images
        doc = self.load_doc(path)
        page = images.open_page(path)
        marked = images.draw_marks(page, [(r.id, r.bbox, "pending") for r in doc.active_regions()])
        url, shown = images.page_url(marked, self.settings.page_image_max_side)
        size = f"This page is {page.width}x{page.height} pixels" + (f" ({shown})." if shown else ".")
        messages = [{"role": "system", "content": RECALL_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": size},
                        {"type": "image_url", "image_url": {"url": url, "detail": self.settings.image_detail}}]}]
        reply = await self.client.complete(messages, [RECALL_TOOL], "required",
                                           effort=self.settings.review_reasoning_effort, cache_key="recall")
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
            self.session.recall_failed = True
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
        url, shown = images.page_url(images.draw_marks(page, self.session.marks()), self.settings.page_image_max_side)
        checked = ", checked again: the list gives the current translations" if doc.revise else ""
        recap = "" if self.settings.whole_chapter_conversation else self.recap(doc.stem)
        text = (f"{recap}Page {index + 1} of {len(self.pages)}: {os.path.basename(doc.image_path)} "
                f"({page.width}x{page.height} px{', ' + shown if shown else ''}), "
                f"{len(regions)} candidate region(s){checked}.\n"
                f"{self.session.candidate_lines()}\n")
        if recalled:
            text += ("A recall check added " + ", ".join(f"#{i}" for i in recalled)
                     + " for text the detector missed; its reading is only a hint. Handle them like the others "
                       "(skip_region not_text if one is not really text).\n")
        if doc.revise:
            text += "Check this page against the image, fix what is wrong with the tools, then call mark_page_done."
        else:
            text += "Work through this page with the tools, then call mark_page_done."
        content = [{"type": "text", "text": text},
                   {"type": "image_url", "image_url": {"url": url, "detail": self.settings.image_detail}}]
        context, urls = self.context_images(doc)
        if urls:
            content.append({"type": "text", "text": context})
            content += [{"type": "image_url", "image_url": {"url": u, "detail": self.settings.image_detail}}
                        for u in urls]
        return {"role": "user", "content": content}

    # --- pages --------------------------------------------------------------
    async def run_page(self, index: int, doc: PageDoc) -> Dict[str, Any]:
        st = self.settings
        if not st.whole_chapter_conversation:
            if self.history:
                self.fresh_page()
        else:
            if self.session.last_prompt_tokens >= st.compact_threshold_tokens:
                self.compact("threshold")
            self.drop_images(index)
        self.session.start_page(index, doc)
        page = doc.stem
        boxes = doc.boxes()
        self.events.emit("page_start", page=page, index=index, total=len(self.pages),
                         regions=len(doc.active_regions()), image=doc.image_path, boxes=boxes)
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
            single = len(reply.tool_calls) == 1
            for call in reply.tool_calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": call.arguments}
                self.events.emit("tool_call", page=page, tool=call.name, args=_compact_args(args),
                                 reason=args.get("reason", "") if isinstance(args, dict) else "")
                if done:  # mark_page_done ended the page: a change after it would come after the checks
                    self.events.emit("tool_result", page=page, tool=call.name, ok=False, summary=NOT_RUN, images=0)
                    self.history.append((index, {"role": "tool", "tool_call_id": call.id,
                                                 "content": "Not run: mark_page_done already finished the page."}))
                    continue
                if call.name == "mark_page_done":
                    self.session.render_issues = await self._render_issues(doc)
                result = self.registry.execute(call.name, call.arguments)
                if not result.ok and call.name != "mark_page_done":
                    self.session.page_tool_errors += 1
                extra = {}
                now = doc.boxes()
                if now != boxes:  # the regions changed: the Agents page redraws its boxes
                    boxes = extra["boxes"] = now
                self.events.emit("tool_result", page=page, tool=call.name, ok=result.ok, summary=result.summary,
                                 images=len(result.images), **extra)
                content: Any = result.text
                if single and result.ok and call.name in BATCHED_SAVES and len(doc.missing()) >= 2:
                    # One save per response costs a request each; ask for the rest in one go.
                    content += BATCH_HINT.format(left=len(doc.missing()))
                elif single and result.ok and call.name in STRUCTURE_EDITS:
                    content += STRUCTURE_HINT
                if result.images:
                    content = [{"type": "text", "text": content}] + [
                        {"type": "image_url", "image_url": {"url": url, "detail": "high"}} for url in result.images]
                self.history.append((index, {"role": "tool", "tool_call_id": call.id, "content": content}))
                done = done or result.page_done
            if done:
                break
        if not done and doc.status != "done":
            doc.status = "incomplete"
            missing = doc.missing()
            reason = f"stopped after {self.session.page_steps} steps; missing " + \
                     (", ".join(f"#{i}" for i in missing) or "nothing")
            if missing:
                reason += " (kept as in the original: their text is not erased)"
            doc.flag(None, "step_limit", reason)
            self.events.emit("flag", page=page, region=None, flag="step_limit", reason=reason)
        return self._finish_page(doc)

    async def _render_issues(self, doc: PageDoc) -> list:
        """``render_check`` issues of the page as the agent is about to finish it; none when a region
        is still without a translation (mark_page_done refuses the page then) or the check fails."""
        if self.render_check is None or doc.missing():
            return []
        try:
            issues = await self.render_check.issues(doc)
        except (AgentStopped, asyncio.CancelledError):
            raise
        except Exception as exc:
            logger.warning("agent: the render size check failed", exc_info=True)
            self.events.emit("render_check", page=doc.stem, ok=False, error=f"{type(exc).__name__}: {exc}"[:200])
            return []
        self.events.emit("render_check", page=doc.stem, ok=True, issues=[i.to_dict() for i in issues])
        return issues

    def _hand_over(self, doc: PageDoc, kind: str, **fields: Any) -> None:
        """The user saved the page in the editor during the run: their version stays (status
        ``edited``) and the review list says so."""
        doc.status = "edited"
        reason = json_guard.reason(self.target_lang, kind, **fields)
        # not one of the model's flag types (page.FLAG_TYPES): only the program sets it
        doc.flags.append({"region": None, "type": "conflict", "reason": reason, "agent": "translator"})
        self.events.emit("flag", page=doc.stem, region=None, flag="conflict", reason=reason)

    def _finish_page(self, doc: PageDoc) -> Dict[str, Any]:
        missing = doc.missing()
        if doc.status == "done" and missing:  # a done page has nothing left without a translation
            doc.status = "incomplete"
            reason = ("marked done with " + ", ".join(f"#{i}" for i in missing) + " still without a translation "
                      "(kept as in the original: their text is not erased)")
            doc.flag(None, "other", reason)
            self.events.emit("flag", page=doc.stem, region=None, flag="other", reason=reason)
        if json_guard.taken(doc.json_path):  # saved in the editor while the agent worked on it
            copy = json_guard.keep_agent_version(doc.json_path, doc.json_data(), self.store.agent_dir)
            self._hand_over(doc, "during", file=copy)
        else:
            doc.save()
            json_guard.note(doc.json_path)
        self.store.save_sidecar(doc.stem, doc.sidecar())
        self.session.finish_page()
        self.done_pages.append(doc.stem)
        outcome = {"page": doc.stem, "image": doc.image_path, "status": doc.status, "confidence": doc.confidence,
                   "flags": len(doc.flags), "missing": doc.missing()}
        self.events.emit("page_done", page=doc.stem, status=doc.status, confidence=doc.confidence,
                         agent_confidence=doc.agent_confidence, check_score=doc.check_score,
                         checks=len(doc.checks), flags=len(doc.flags), missing=doc.missing(),
                         memory=self.memory.counts(), image=doc.image_path, boxes=doc.boxes())
        return outcome

    async def run(self) -> List[Dict[str, Any]]:
        self.events.emit("run_start", pages=[os.path.basename(p) for p in self.pages],
                         stems=[page_stem(p) for p in self.pages],
                         model=getattr(self.client, "model_label", ""), target_lang=self.target_lang,
                         memory=self.memory.counts(),
                         work_prompt={"name": self.work_prompt.name, "terms": len(self.work_prompt.terms)}
                         if self.work_prompt else None,
                         fonts=list(self.session.font_choices) or None, **({"revise": True} if self.revise else {}),
                         **self.run_info)
        outcomes: List[Dict[str, Any]] = []
        failures_in_a_row = 0
        ended = ""  # why the run ended early: "stopped" or "failed"
        try:
            for index, path in enumerate(self.pages):
                if self.should_stop():
                    raise AgentStopped()
                stem = page_stem(path)
                failed = self.preprocess_failed.get(os.path.normcase(os.path.abspath(path)))
                if failed:  # its JSON is the one from before the run: no model call for it
                    error = f"preprocess failed: {failed}"
                    self.events.emit("error", page=stem, error=error, stage="preprocess", image=path)
                    outcomes.append({"page": stem, "image": path, "status": "error", "error": error})
                    continue
                try:
                    doc = self.load_doc(path)
                except Exception as exc:
                    self.events.emit("error", page=stem, error=f"cannot open page data: {exc}")
                    outcomes.append({"page": stem, "image": path, "status": "error", "error": str(exc)})
                    continue
                if json_guard.taken(doc.json_path):  # saved in the editor before the agent got to it
                    self._hand_over(doc, "later")
                    outcome = {"page": stem, "image": path, "status": "edited", "confidence": None,
                               "flags": len(doc.flags), "missing": []}
                    self.events.emit("page_done", page=stem, status="edited", flags=len(doc.flags), missing=[],
                                     image=path, boxes=doc.boxes())
                    outcomes.append(outcome)
                    if self.on_page_done is not None:
                        self.on_page_done(outcome)
                    continue
                json_guard.note(doc.json_path)
                try:
                    outcome = await self.run_page(index, doc)
                    outcomes.append(outcome)
                    failures_in_a_row = 0
                    if self.on_page_done is not None:
                        self.on_page_done(outcome)
                except (AgentStopped, asyncio.CancelledError):
                    # the stop button cancels the task: the page keeps what it has, marked stopped
                    ended = "stopped"
                    doc.status = "stopped"
                    try:
                        self._finish_page(doc)
                    except Exception:
                        logger.exception(f"saving the stopped page {stem} failed")
                    raise
                except Exception as exc:
                    logger.exception(f"agent page {stem} failed")
                    failures_in_a_row += 1
                    doc.status = "error"
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    doc.flag(None, "error", error)
                    self.events.emit("error", page=stem, error=error)
                    outcomes.append({**self._finish_page(doc), "error": error})
                    # The failed page's half-finished turn must not leak into the next page.
                    self.history = [(i, m) for i, m in self.history if i != index]
                    if failures_in_a_row >= 3:
                        self.events.emit("error", error="three pages failed in a row; stopping the run")
                        ended = "failed"
                        break
        except (AgentStopped, asyncio.CancelledError):
            ended = "stopped"  # also between two pages
            raise
        finally:
            pending = list(self._recall_tasks.values())
            self._recall_tasks.clear()
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            self.events.emit("run_end", pages=len(outcomes), total=len(self.pages), reason=ended,
                             done=sum(1 for o in outcomes if o.get("status") == "done"),
                             flagged=sum(o.get("flags", 0) for o in outcomes), totals=self.totals,
                             memory=self.memory.counts())
        return outcomes


NOT_RUN = "not run: the page was already done"
"""The timeline's result of a call that came after mark_page_done in the same response."""


def _compact_args(args: Any, limit: int = 300) -> Any:
    """Arguments for the event log, with long strings shortened."""
    if isinstance(args, dict):
        return {k: _compact_args(v, limit) for k, v in args.items()}
    if isinstance(args, list):
        return [_compact_args(v, limit) for v in args[:50]]
    if isinstance(args, str) and len(args) > limit:
        return args[: limit - 1] + "…"
    return args
