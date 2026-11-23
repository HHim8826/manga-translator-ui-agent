"""Program checks of a finished page and the page confidence built from them.

The model's own confidence is not usable on its own (M2: 0.98-0.99 on every
page, also on pages with missed text), so the confidence of a page is the
lower of the model's number and a score computed here from checks the program
can do reliably:

* ``untranslated_kana``  - Japanese kana left in the translation
* ``glossary_mismatch``  - a glossary term is in the source, its translation is not
* ``character_mismatch`` - same for a character name
* ``empty_source``       - a translation without a source transcription
* ``overflow``           - far more characters than the box holds at the original size
* ``needs_second_read``  - small, handwritten or added text, or text the local
  OCR read differently, that was never zoomed
* ``missing_line_break`` - AI line breaks are on and a source of two or more
  lines has a translation without [BR], so it would be typeset as one line
* ``line_overflow``      - AI line breaks are on and a line is far longer than the box (or there are
  far more lines than fit across it): the renderer follows [BR] and does not shrink the text, so it
  runs out of the box

The first ``mark_page_done`` returns these issues to the model so it can fix
them. What is left when the page is finished lowers the score, and the
serious ones go to the review list.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from .page import BREAK, needs_break, text_length
from .store import normalize

if TYPE_CHECKING:  # pragma: no cover
    from .page import Region
    from .tools import AgentSession

KANA = re.compile(r"[\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fd-\u30ff\uff66-\uff6f\uff71-\uff9d]+")
"""Hiragana and katakana, without the long-vowel mark ー and the middle dot ・
that Chinese text uses too."""

WEIGHTS = {
    "untranslated_kana": 0.5,
    "glossary_mismatch": 0.35,
    "character_mismatch": 0.35,
    "empty_source": 0.3,
    "overflow": 0.15,
    "needs_second_read": 0.15,
    "missing_line_break": 0.1,
    "line_overflow": 0.15,
}
SERIOUS = {"untranslated_kana": "uncertain_translation", "glossary_mismatch": "uncertain_translation",
           "character_mismatch": "uncertain_translation", "empty_source": "uncertain_ocr"}
"""Issues that go to the review list when they are still there at the end, with their flag type."""

REGION_FLAG_WEIGHT = 0.25
PAGE_FLAG_WEIGHT = 0.2
SMALL_FONT_PX = 22
OVERFLOW_RATIO = 2.5
LINE_ROOM_RATIO = 1.2
LINE_ROOM_SLACK = 1.0
"""A line may run this far past the box: ``ratio * room + slack`` cells (bubbles have room around the
text box, most of all around short lines). Tuned on chapters 27-36: of 1167 regions the user accepted,
6 go past it; the 6 regions of the ch34 check that ran far out of their box (1.4 to 2 times its
length) all do."""


@dataclass
class Issue:
    code: str
    region: Optional[int]
    detail: Dict[str, object]
    weight: float

    def to_dict(self) -> dict:
        return asdict(self)


# --- texts ----------------------------------------------------------------------
AGENT_TEXT = {
    "untranslated_kana": "#{region}: the translation still contains Japanese kana 「{kana}」.",
    "glossary_mismatch": "#{region}: the source has 「{source}」, the glossary translates it as 「{target}」, "
                         "but the translation does not use that.",
    "character_mismatch": "#{region}: the source has the character 「{source}」, the character list says "
                          "「{target}」, but the translation does not use that.",
    "empty_source": "#{region}: no source transcription was saved.",
    "overflow": "#{region}: the translation has {length} characters, the box holds about {capacity} at the "
                "original size; shorten it if you can.",
    "needs_second_read": "#{region}: {why}, and it was not zoomed. Zoom for a second read and fix set_text if "
                         "the reading changes.",
    "missing_line_break": "#{region}: the source has {lines} lines but the translation has no [BR], so it would "
                          "be typeset as one line. Add [BR] at natural breaks with set_line_breaks.",
    "line_overflow": "#{region}: {lines} line(s), the longest {longest} characters, but at its font size the box "
                     "fits about {room} characters per line and {rows} line(s). The renderer keeps your [BR] and "
                     "does not shrink the text, so it would run out of the box: break it into shorter lines of "
                     "about even length (set_line_breaks), or shorten it.",
}
WHY = {
    "small": "small text",
    "handwritten": "handwritten text",
    "added": "a region added for text the detector missed",
    "ocr_differs": "your reading differs from the local OCR 「{local}」",
    "unsure": "you were not sure of it",
}

HUMAN_TEXT = {
    "CHT": {
        "untranslated_kana": "譯文裡還有日文假名「{kana}」",
        "glossary_mismatch": "原文有「{source}」，術語表譯為「{target}」，但譯文沒有使用",
        "character_mismatch": "原文有角色「{source}」，角色表譯名是「{target}」，但譯文沒有使用",
        "empty_source": "沒有記錄原文",
        "overflow": "譯文 {length} 字，框內以原字級約容納 {capacity} 字",
        "needs_second_read": "沒有放大確認",
        "missing_line_break": "原文 {lines} 行，譯文沒有換行",
        "line_overflow": "最長一行 {longest} 字，框內以原字級每行約容納 {room} 字，會超出框",
        "page": "程式檢查的信心分數 {score:.2f}：{summary}",
        "count": "{count} 個框{label}",
        "short": {"untranslated_kana": "留有假名", "glossary_mismatch": "沒用術語表譯法",
                  "character_mismatch": "沒用角色譯名", "empty_source": "沒有原文", "overflow": "譯文過長",
                  "needs_second_read": "沒有放大確認", "missing_line_break": "沒有換行",
                  "line_overflow": "超出框"},
        "none": "沒有發現問題",
    },
    "CHS": {
        "untranslated_kana": "译文里还有日文假名「{kana}」",
        "glossary_mismatch": "原文有「{source}」，术语表译为「{target}」，但译文没有使用",
        "character_mismatch": "原文有角色「{source}」，角色表译名是「{target}」，但译文没有使用",
        "empty_source": "没有记录原文",
        "overflow": "译文 {length} 字，框内以原字号约容纳 {capacity} 字",
        "needs_second_read": "没有放大确认",
        "missing_line_break": "原文 {lines} 行，译文没有换行",
        "line_overflow": "最长一行 {longest} 字，框内以原字号每行约容纳 {room} 字，会超出框",
        "page": "程序检查的置信分数 {score:.2f}：{summary}",
        "count": "{count} 个框{label}",
        "short": {"untranslated_kana": "留有假名", "glossary_mismatch": "没用术语表译法",
                  "character_mismatch": "没用角色译名", "empty_source": "没有原文", "overflow": "译文过长",
                  "needs_second_read": "没有放大确认", "missing_line_break": "没有换行",
                  "line_overflow": "超出框"},
        "none": "没有发现问题",
    },
    "ENG": {
        "untranslated_kana": "the translation still contains Japanese kana 「{kana}」",
        "glossary_mismatch": "the source has 「{source}」 (glossary: 「{target}」) but the translation does not use it",
        "character_mismatch": "the source has 「{source}」 (character list: 「{target}」) but the translation does not use it",
        "empty_source": "no source text was saved",
        "overflow": "{length} characters for room of about {capacity} at the original size",
        "needs_second_read": "not zoomed for a second read",
        "missing_line_break": "the source has {lines} lines, the translation has no line break",
        "line_overflow": "the longest line has {longest} characters, the box fits about {room} per line at the "
                         "original size, so it runs out of the box",
        "page": "program check score {score:.2f}: {summary}",
        "count": "{count} region(s): {label}",
        "short": {"untranslated_kana": "kana left", "glossary_mismatch": "glossary not used",
                  "character_mismatch": "character name not used", "empty_source": "no source text",
                  "overflow": "translation too long", "needs_second_read": "not zoomed",
                  "missing_line_break": "no line break", "line_overflow": "runs out of the box"},
        "none": "no issues found",
    },
}


def human_text(lang: str, key: str, **fields) -> str:
    table = HUMAN_TEXT.get(lang) or HUMAN_TEXT["ENG"]
    return table[key].format(**fields)


def human_language(target_lang: str) -> str:
    """Program-written review texts follow the target language like the agent's reasons."""
    return target_lang if target_lang in HUMAN_TEXT else "ENG"


def agent_text(issue: Issue) -> str:
    detail = dict(issue.detail)
    if issue.code == "needs_second_read":
        detail["why"] = "; ".join(WHY[w].format(**detail) for w in detail.get("why", []))
    return AGENT_TEXT[issue.code].format(region=issue.region, **detail)


# --- checks ---------------------------------------------------------------------
def _alternatives(target: str) -> List[str]:
    """``小迪梅（暱稱）`` / ``A/B``: any of the written forms counts."""
    plain = re.sub(r"[（(][^）)]*[）)]", "", target)
    parts = re.split(r"[/／|｜,，、]", plain)
    return [normalize(p) for p in parts if normalize(p)]


def _work_terms_in(work_terms, src_norm: str) -> list:
    """Terms of the work's prompt file found in the source; a term inside a longer one that is also
    there (みゃー in みゃーさん) is left to the longer one."""
    found = {}
    for term in work_terms:
        key = normalize(term.source)
        if len(key) >= 2 and key in src_norm:
            found.setdefault(key, term)
    return [term for key, term in found.items()
            if not any(key != other and key in other for other in found)]


def term_issues(region_id: int, source: str, translation: str, work_terms, memory) -> List[Issue]:
    """Glossary and character names: a term in the source whose translation is not used. The work's
    prompt file comes first; the agent's own glossary does not repeat a term it already checked."""
    issues: List[Issue] = []
    src_norm, tr_norm = normalize(source), normalize(str(translation or "").replace(BREAK, ""))
    if not src_norm:
        return issues
    checked = set()
    for term in _work_terms_in(work_terms, src_norm):
        checked.add(normalize(term.source))
        alternatives = [a for target in term.targets for a in _alternatives(target)]
        if alternatives and not any(a in tr_norm for a in alternatives):
            issues.append(Issue("glossary_mismatch", region_id,
                                {"source": term.source, "target": " / ".join(term.targets)},
                                WEIGHTS["glossary_mismatch"]))
    for entry in memory.glossary:
        term = normalize(entry.get("source", ""))
        if term in checked:
            continue
        alternatives = _alternatives(entry.get("target", ""))
        if len(term) >= 2 and term in src_norm and alternatives and not any(a in tr_norm for a in alternatives):
            issues.append(Issue("glossary_mismatch", region_id, {"source": entry["source"], "target": entry["target"]},
                                WEIGHTS["glossary_mismatch"]))
    for name, info in memory.characters.items():
        term = normalize(name)
        alternatives = _alternatives(info.get("target_name", ""))
        if len(term) >= 2 and term in src_norm and alternatives and not any(a in tr_norm for a in alternatives):
            issues.append(Issue("character_mismatch", region_id,
                                {"source": name, "target": info.get("target_name", "")},
                                WEIGHTS["character_mismatch"]))
    return issues


def _capacity(region: "Region") -> int:
    x1, y1, x2, y2 = region.bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    fs = max(8, region.font_size or int(min(w, h) * 0.8))
    if region.direction.startswith("v"):
        return max(1, int(w // (fs * 1.1))) * max(1, int(h // fs))
    return max(1, int(h // (fs * 1.2))) * max(1, int(w // fs))


def line_room(region: "Region") -> Tuple[float, float]:
    """Characters per line and lines that fit the box at the region's font size (the length along the
    text and across it, in font sizes)."""
    x1, y1, x2, y2 = region.bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    fs = max(8, region.font_size or int(min(w, h) * 0.8))
    if region.direction.startswith("v"):
        room, rows = h / fs, w / (fs * 1.1)
    else:
        room, rows = w / fs, h / (fs * 1.2)
    # the source's own lines fit the box at this size: ch34 two-column boxes measured 1.8 lines (shown as
    # "1 line"), and the proofreader would not break lines that the source line count asks for
    return room, max(rows, float(region.line_count))


def room_counts(region: "Region") -> Tuple[int, int]:
    """``line_room`` as the agents read it: characters per line and lines, rounded, at least 1."""
    room, rows = line_room(region)
    return max(1, int(room + 0.5)), max(1, int(rows + 0.5))


def _cells(line: str) -> float:
    """Length of a line in font sizes: a CJK character is one, punctuation and Latin letters or digits
    about half (they are narrow, or the renderer squeezes them)."""
    return sum(1.0 if ch.isalnum() and not ch.isascii() else 0.5 for ch in re.sub(r"\s", "", line))


def line_overflow(region: "Region", translation: str) -> Optional[Dict[str, int]]:
    """With AI line breaks: the details of a translation whose lines run out of the box, else None."""
    lines = str(translation or "").split(BREAK)
    longest = max(lines, key=_cells, default="")
    room, rows = line_room(region)
    too_long = _cells(longest) > room * LINE_ROOM_RATIO + LINE_ROOM_SLACK
    if too_long or len(lines) > max(rows * LINE_ROOM_RATIO, rows + 1):
        shown_room, shown_rows = room_counts(region)
        return {"lines": len(lines), "longest": len(re.sub(r"\s", "", longest)), "room": shown_room,
                "rows": shown_rows}
    return None


def region_issues(session: "AgentSession", region: "Region") -> List[Issue]:
    issues: List[Issue] = []
    translation = region.translation or ""
    source = region.source_text or ""
    target_lang = getattr(session, "target_lang", "") or ""

    if target_lang != "JPN":
        kana = KANA.findall(translation)
        if kana:
            issues.append(Issue("untranslated_kana", region.id, {"kana": "".join(kana)[:20]},
                                WEIGHTS["untranslated_kana"]))

    src_norm = normalize(source)
    if src_norm:
        issues.extend(term_issues(region.id, source, translation, getattr(session, "work_terms", None) or [],
                                  session.memory))
    else:
        issues.append(Issue("empty_source", region.id, {}, WEIGHTS["empty_source"]))

    length = text_length(translation)
    capacity = _capacity(region)
    if length > OVERFLOW_RATIO * capacity:
        issues.append(Issue("overflow", region.id, {"length": length, "capacity": capacity}, WEIGHTS["overflow"]))

    if getattr(session, "line_breaks", False) and needs_break(region.line_count, translation):
        issues.append(Issue("missing_line_break", region.id, {"lines": region.line_count},
                            WEIGHTS["missing_line_break"]))
    elif getattr(session, "line_breaks", False):
        over = line_overflow(region, translation)
        if over:
            issues.append(Issue("line_overflow", region.id, over, WEIGHTS["line_overflow"]))

    mode = session.settings.double_read
    # a translation kept from the earlier run had its reads then (a page checked again)
    if mode != "none" and not region.unchanged and not session.was_zoomed(region):
        why: List[str] = []
        if mode == "all":
            why.append("unsure")
        if region.category == "handwritten":
            why.append("handwritten")
        if region.font_size and region.font_size < SMALL_FONT_PX:
            why.append("small")
        if region.origin in ("recall", "agent", "cut"):
            why.append("added")
        if region.confidence is not None and region.confidence < 0.8 and "unsure" not in why:
            why.append("unsure")
        local = normalize(region.ocr_local)
        if region.origin == "detector" and src_norm and \
                difflib.SequenceMatcher(None, local, src_norm).ratio() < 0.5:
            why.append("ocr_differs")
        if why:
            issues.append(Issue("needs_second_read", region.id,
                                {"why": why, "local": (region.ocr_local or "").replace("\n", " ")[:30]},
                                WEIGHTS["needs_second_read"]))
    return issues


def page_issues(session: "AgentSession") -> List[Issue]:
    """Issues of the translated regions; on a page checked again, not of the user's locked regions."""
    issues: List[Issue] = []
    for region in session.doc.active_regions():
        if region.translation and not region.skipped and not region.locked:
            issues.extend(region_issues(session, region))
    return issues


def page_score(session: "AgentSession", issues: List[Issue]) -> float:
    """Lowest region score times the page-level factors, 0-1."""
    doc = session.doc
    scores: Dict[int, float] = {}
    for region in doc.active_regions():
        if region.skipped or not region.translation or region.locked:
            continue
        score = 1.0
        if region.confidence is not None:
            score *= 1.0 - max(0.0, 0.9 - region.confidence)
        scores[region.id] = score
    for issue in issues:
        if issue.region in scores:
            scores[issue.region] *= 1.0 - issue.weight
    for flag in doc.flags:
        if flag.get("region") in scores:
            scores[flag["region"]] *= 1.0 - REGION_FLAG_WEIGHT
    score = min(scores.values()) if scores else 1.0
    for flag in doc.flags:
        if flag.get("region") is None and flag.get("type") != "low_confidence":
            score *= 1.0 - PAGE_FLAG_WEIGHT
    if getattr(session, "recall_failed", False):
        score *= 0.9
    if getattr(session, "page_tool_errors", 0) >= 3:
        score *= 0.9
    if session.page_steps > 0.75 * session.settings.max_steps_per_page:
        score *= 0.9
    return round(max(0.0, min(1.0, score)), 3)


def _counts(issues: List[Issue]) -> Dict[str, int]:
    """Regions per check, in the order the checks were found."""
    counts: Dict[str, set] = {}
    for issue in issues:
        counts.setdefault(issue.code, set()).add(issue.region)
    return {code: len(regions) for code, regions in counts.items()}


def codes(issues: List[Issue]) -> str:
    """``checks untranslated_kana:1, overflow:2`` for the run log; the AI Agent page says it in the
    language of the app (the review list uses ``summary``, in the target language)."""
    return "checks " + ", ".join(f"{code}:{count}" for code, count in _counts(issues).items())


def summary(lang: str, issues: List[Issue]) -> str:
    table = HUMAN_TEXT.get(lang) or HUMAN_TEXT["ENG"]
    parts = [table["count"].format(count=count, label=table["short"][code]) for code, count in _counts(issues).items()]
    if not parts:
        return table["none"]
    return "；".join(parts) if lang in ("CHT", "CHS") else "; ".join(parts)
