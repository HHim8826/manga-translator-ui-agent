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

The first ``mark_page_done`` returns these issues to the model so it can fix
them. What is left when the page is finished lowers the score, and the
serious ones go to the review list.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

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
}
SERIOUS = {"untranslated_kana": "uncertain_translation", "glossary_mismatch": "uncertain_translation",
           "character_mismatch": "uncertain_translation", "empty_source": "uncertain_ocr"}
"""Issues that go to the review list when they are still there at the end, with their flag type."""

REGION_FLAG_WEIGHT = 0.25
PAGE_FLAG_WEIGHT = 0.2
SMALL_FONT_PX = 22
OVERFLOW_RATIO = 2.5


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
        "page": "程式檢查的信心分數 {score:.2f}：{summary}",
        "count": "{count} 個框{label}",
        "short": {"untranslated_kana": "留有假名", "glossary_mismatch": "沒用術語表譯法",
                  "character_mismatch": "沒用角色譯名", "empty_source": "沒有原文", "overflow": "譯文過長",
                  "needs_second_read": "沒有放大確認"},
        "none": "沒有發現問題",
    },
    "CHS": {
        "untranslated_kana": "译文里还有日文假名「{kana}」",
        "glossary_mismatch": "原文有「{source}」，术语表译为「{target}」，但译文没有使用",
        "character_mismatch": "原文有角色「{source}」，角色表译名是「{target}」，但译文没有使用",
        "empty_source": "没有记录原文",
        "overflow": "译文 {length} 字，框内以原字号约容纳 {capacity} 字",
        "needs_second_read": "没有放大确认",
        "page": "程序检查的置信分数 {score:.2f}：{summary}",
        "count": "{count} 个框{label}",
        "short": {"untranslated_kana": "留有假名", "glossary_mismatch": "没用术语表译法",
                  "character_mismatch": "没用角色译名", "empty_source": "没有原文", "overflow": "译文过长",
                  "needs_second_read": "没有放大确认"},
        "none": "没有发现问题",
    },
    "ENG": {
        "untranslated_kana": "the translation still contains Japanese kana 「{kana}」",
        "glossary_mismatch": "the source has 「{source}」 (glossary: 「{target}」) but the translation does not use it",
        "character_mismatch": "the source has 「{source}」 (character list: 「{target}」) but the translation does not use it",
        "empty_source": "no source text was saved",
        "overflow": "{length} characters for room of about {capacity} at the original size",
        "needs_second_read": "not zoomed for a second read",
        "page": "program check score {score:.2f}: {summary}",
        "count": "{count} region(s): {label}",
        "short": {"untranslated_kana": "kana left", "glossary_mismatch": "glossary not used",
                  "character_mismatch": "character name not used", "empty_source": "no source text",
                  "overflow": "translation too long", "needs_second_read": "not zoomed"},
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


def _work_terms_in(session: "AgentSession", src_norm: str) -> list:
    """Terms of the work's prompt file found in the source; a term inside a longer one that is also
    there (みゃー in みゃーさん) is left to the longer one."""
    found = {}
    for term in getattr(session, "work_terms", None) or []:
        key = normalize(term.source)
        if len(key) >= 2 and key in src_norm:
            found.setdefault(key, term)
    return [term for key, term in found.items()
            if not any(key != other and key in other for other in found)]


def _capacity(region: "Region") -> int:
    x1, y1, x2, y2 = region.bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    fs = max(8, region.font_size or int(min(w, h) * 0.8))
    if region.direction.startswith("v"):
        return max(1, int(w // (fs * 1.1))) * max(1, int(h // fs))
    return max(1, int(h // (fs * 1.2))) * max(1, int(w // fs))


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

    src_norm, tr_norm = normalize(source), normalize(translation)
    if src_norm:
        checked = set()
        for term in _work_terms_in(session, src_norm):  # the work's prompt file comes first
            checked.add(normalize(term.source))
            alternatives = [a for target in term.targets for a in _alternatives(target)]
            if alternatives and not any(a in tr_norm for a in alternatives):
                issues.append(Issue("glossary_mismatch", region.id,
                                    {"source": term.source, "target": " / ".join(term.targets)},
                                    WEIGHTS["glossary_mismatch"]))
        for entry in session.memory.glossary:
            if normalize(entry.get("source", "")) in checked:
                continue
            term = normalize(entry.get("source", ""))
            alternatives = _alternatives(entry.get("target", ""))
            if len(term) >= 2 and term in src_norm and alternatives and not any(a in tr_norm for a in alternatives):
                issues.append(Issue("glossary_mismatch", region.id,
                                    {"source": entry["source"], "target": entry["target"]},
                                    WEIGHTS["glossary_mismatch"]))
        for name, info in session.memory.characters.items():
            term = normalize(name)
            alternatives = _alternatives(info.get("target_name", ""))
            if len(term) >= 2 and term in src_norm and alternatives and not any(a in tr_norm for a in alternatives):
                issues.append(Issue("character_mismatch", region.id,
                                    {"source": name, "target": info.get("target_name", "")},
                                    WEIGHTS["character_mismatch"]))
    else:
        issues.append(Issue("empty_source", region.id, {}, WEIGHTS["empty_source"]))

    length = sum(1 for ch in translation if ch.isalnum())   # punctuation squeezes in; 「不行。」 is 2
    capacity = _capacity(region)
    if length > OVERFLOW_RATIO * capacity:
        issues.append(Issue("overflow", region.id, {"length": length, "capacity": capacity}, WEIGHTS["overflow"]))

    mode = session.settings.double_read
    if mode != "none" and not session.was_zoomed(region):
        why: List[str] = []
        if mode == "all":
            why.append("unsure")
        if region.category == "handwritten":
            why.append("handwritten")
        if region.font_size and region.font_size < SMALL_FONT_PX:
            why.append("small")
        if region.origin in ("recall", "agent"):
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
    issues: List[Issue] = []
    for region in session.doc.active_regions():
        if region.translation and not region.skipped:
            issues.extend(region_issues(session, region))
    return issues


def page_score(session: "AgentSession", issues: List[Issue]) -> float:
    """Lowest region score times the page-level factors, 0-1."""
    doc = session.doc
    scores: Dict[int, float] = {}
    for region in doc.active_regions():
        if region.skipped or not region.translation:
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


def summary(lang: str, issues: List[Issue]) -> str:
    table = HUMAN_TEXT.get(lang) or HUMAN_TEXT["ENG"]
    counts: Dict[str, set] = {}
    for issue in issues:
        counts.setdefault(issue.code, set()).add(issue.region)
    parts = [table["count"].format(count=len(regions), label=table["short"][code]) for code, regions in counts.items()]
    if not parts:
        return table["none"]
    return "；".join(parts) if lang in ("CHT", "CHS") else "; ".join(parts)
