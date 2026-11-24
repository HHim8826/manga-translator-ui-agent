"""Tool results on the AI Agent page, in the interface language.

The agent tools write a short summary of each result into the event log (``tool_result``) in fixed
forms such as ``confidence 0.99`` or ``#10 a2 lama_large: ok``. This module turns the known forms
into localized text, so old run logs read the same way. Text the model or the user wrote (source
text, translations, query words) stays as it is. A failed call shows a short localized note; its
message, written for the model, goes to the tooltip.
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Tuple

Translate = Callable[..., str]

SKIP_KINDS = ("sign", "sfx", "credit", "not_text", "duplicate", "other")
FLAG_KINDS = ("uncertain_ocr", "uncertain_translation", "layout", "inpaint", "missing_text", "low_confidence",
              "step_limit", "error", "other", "conflict")
ISSUES = ("residual_text", "broken_lines")
CHECKS = ("untranslated_kana", "glossary_mismatch", "character_mismatch", "empty_source", "overflow",
          "needs_second_read", "missing_line_break", "line_overflow", "overlap", "drawn_large", "drawn_small",
          "drawn_unlike", "drawn_turned")
EDITOR_ORDER = ("proofread", "fonts", "editor")
"""Who numbers regions like the editor (the JSON order): the proofreader, the font check, and the
editor's AI Agent tab. The translator's own numbers (#4) are the preview's boxes."""

# the first word of a translator undo label (page.py _checkpoint) → the tool it undid
_UNDO_TOOLS = {"set_text": "set_text", "skip": "skip_region", "merge": "merge_regions", "split": "split_region",
               "set_reading_order": "set_reading_order", "link": "link_regions", "line_breaks": "set_line_breaks",
               "flag": "flag_for_review", "add_region": "add_region"}

_ID = r"(#\d+)"


def _known(t: Translate, prefix: str, value: str, choices) -> str:
    return t(f"{prefix} {value}") if value in choices else value


def flag_kind(t: Translate, kind: str) -> str:
    return _known(t, "Agent Flag", kind, FLAG_KINDS)


def region_label(t: Translate, region, agent: str = "translator") -> str:
    """``#4`` (the translator's region), ``第 4 框`` (the editor's box) or ``區域 4`` (an inpaint area)."""
    if agent in EDITOR_ORDER:
        return t("Agent Result Editor Box", n=region)
    if agent == "inpaint":
        return t("Agent Review Area", id=region)
    return f"#{region}"


def flag_text(t: Translate, region, kind: str, agent: str = "translator") -> str:
    """``#4：原文不確定`` for a flag on region 4 (numbered the agent's way); the kind alone for a page flag."""
    name = flag_kind(t, str(kind or ""))
    return t("Agent Result With Kind", id=region_label(t, region, agent), kind=name) if region else name


def method_name(t: Translate, method: str) -> str:
    """An inpaint method as the human reads it: the model's name, or "fill" (flat colour)."""
    return t("Agent Method fill") if method == "fill" else (method or "-")


def mask_op(t: Translate, label: str) -> str:
    """A mask edit of the inpaint review (inpaint.py labels)."""
    m = re.fullmatch(r"(dilate|erode) (\d+)px( \(residual\))?", label)
    if m:
        return t(f"Agent Mask {m[1]}", px=m[2]) + (t("Agent Mask Residual Only") if m[3] else "")
    m = re.fullmatch(r"paint (add|remove)(?: (\S+))?", label)
    if m:
        return t(f"Agent Mask paint {m[1]}", cells=m[2] or "")
    m = re.fullmatch(r"select (.+)", label)
    if m:
        return t("Agent Mask select", parts=m[1])
    m = re.fullmatch(r"protect lines (\d+)px", label)
    if m:
        return t("Agent Mask protect lines", px=m[1])
    return label


def _undo_label(t: Translate, label: str, tool_keys: dict) -> str:
    m = re.fullmatch(r"#(\d+) (.+)", label)  # inpaint review: "#10 erode 3px"
    if m:
        return f"#{m[1]} {mask_op(t, m[2])}"
    head, _, rest = label.partition(" ")
    tool = _UNDO_TOOLS.get(head)
    if tool is None or tool not in tool_keys:
        return label
    return (t(tool_keys[tool]) + (f" {rest}" if rest else "")).strip()


def _ids(t: Translate, text: str) -> str:
    return t("Agent Result Separator").join(part.strip() for part in text.split(","))


def result_text(t: Translate, tool: str, summary: str, ok: bool = True,
                tool_keys: Optional[dict] = None) -> Tuple[str, str]:
    """(text for the result column, tooltip with the original summary)."""
    summary = str(summary or "")
    return _format(t, tool, summary, ok, tool_keys or {}), summary


def _format(t: Translate, tool: str, s: str, ok: bool, tool_keys: dict) -> str:
    if not ok:
        if s == "not run: the page was already done":  # after mark_page_done in the same response
            return t("Agent Result Not Run")
        m = re.fullmatch(r"missing (#\d+(?:, #\d+)*)", s)
        if tool == "mark_page_done" and m:
            return t("Agent Result Missing", ids=_ids(t, m[1]))
        m = re.fullmatch(r"residual (#\d+(?:, #\d+)*)", s)
        if tool == "finish_review" and m:
            return t("Agent Result Residual", ids=_ids(t, m[1]))
        m = re.fullmatch(r"checks (\w+:\d+(?:, \w+:\d+)*)", s)
        if tool == "mark_page_done" and m:  # the review round of the program checks
            found = t("Agent Result Separator").join(
                t("Agent Check Count", count=n, check=_known(t, "Agent Check", code, CHECKS))
                for code, n in re.findall(r"(\w+):(\d+)", m[1]))
            return t("Agent Result Checks", found=found)
        if tool == "mark_page_done" and s:
            return s  # an older run log: summarized in the target language
        m = re.match(_ID + r" used its (\d+) operations", s)
        if m:  # the inpaint review's per-area limit
            return t("Agent Result Out Of Operations", id=m[1], n=m[2])
        return t("Agent Result Failed")

    if tool == "get_candidates":
        m = re.fullmatch(r"(\d+) regions", s)
        return t("Agent Result Regions", count=m[1]) if m else s
    if tool == "zoom":
        m = re.fullmatch(r"area \[(.*)\] ×(.+)", s)
        return t("Agent Result Zoom Area", box=m[1], scale=m[2]) if m else s
    if tool == "get_neighbors":
        if s == "none":
            return t("Agent Result No Neighbors")
        parts = re.findall(r"(previous|next) page \(([+-]\d+)\)", s)
        if not parts:
            return s
        return t("Agent Result Separator").join(
            t("Agent Result Page Before" if where == "previous" else "Agent Result Page After",
              n=abs(int(offset))) for where, offset in parts)
    if tool == "skip_region":
        m = re.fullmatch(_ID + r" \((\w+)\)", s)
        return t("Agent Result With Kind", id=m[1], kind=_known(t, "Agent Skip", m[2], SKIP_KINDS)) if m else s
    if tool in ("glossary_search", "search_past_translations", "character_get"):
        m = re.fullmatch(r"(.*): (\d+)", s, re.S)
        if not m:
            return s
        query = t("Agent Result All") if tool == "character_get" and m[1] == "all" else m[1]
        return t("Agent Result Found", query=query, count=m[2])
    if tool in ("glossary_upsert", "character_upsert", "set_font_style"):
        m = re.fullmatch(r"(.*) \((added|updated)\)", s, re.S)
        return t("Agent Result Added" if m[2] == "added" else "Agent Result Updated", item=m[1]) if m else s
    if tool in ("story_summary_get", "story_summary_update"):
        m = re.fullmatch(r"(\d+) chars", s)
        return t("Agent Result Chars", count=m[1]) if m else s
    if tool == "flag_for_review":
        m = re.fullmatch(r"(?:" + _ID + r" )?([a-z_]+)", s)
        if m:
            return flag_text(t, m[1][1:] if m[1] else None, m[2])
    if tool == "mark_page_done":
        m = re.fullmatch(r"confidence ([\d.]+)", s)
        return t("Agent Result Confidence", percent=round(float(m[1]) * 100)) if m else s
    if tool == "budget_status":
        m = re.fullmatch(r"(\d+) tokens", s)
        return t("Agent Result Tokens", count=m[1]) if m else s
    if tool == "undo":
        return t("Agent Result Undid", what=_undo_label(t, s, tool_keys))
    if tool == "view_region":
        m = re.fullmatch(_ID + r" a(\d+)", s)
        if m:
            return t("Agent Result Attempt", id=m[1], n=m[2])
        m = re.fullmatch(_ID + r" M(\d+) S(\d+)", s)
        return t("Agent Result Components", id=m[1], masks=m[2], strokes=m[3]) if m else s
    if tool in ("mask_select", "adjust_mask", "mask_paint", "protect_lines"):
        m = re.fullmatch(_ID + r" (.+)", s)
        return f"{m[1]} {mask_op(t, m[2])}" if m else s
    if tool == "inpaint_region":
        m = re.fullmatch(_ID + r" a(\d+) (\S+): (.+)", s)
        if not m:
            return s
        issues = t("Agent Result No Issues") if m[4] == "ok" else t("Agent Result Separator").join(
            _known(t, "Agent Issue", issue.strip(), ISSUES) for issue in m[4].split(","))
        method = method_name(t, m[3])
        return t("Agent Result Inpainted", id=m[1], n=m[2], method=method, issues=issues)
    if tool == "pick_candidate":
        m = re.fullmatch(_ID + r" a(\d+)", s)
        return t("Agent Result Picked", id=m[1], n=m[2]) if m else s
    if tool == "finish_review" and s == "done":
        return t("Agent Result Done")
    if tool == "revise_text" and s == "unchanged":
        return t("Agent Result Unchanged")
    if tool in ("revise_text", "flag_for_review"):
        m = re.fullmatch(r"(\S+) #(\d+)(?:: (.*))?", s, re.S)
        if m:  # the proofreader's "page #id": numbered like the editor's boxes, not the preview's
            box = f"{m[1]} {region_label(t, m[2], 'proofread')}"
            return f"{box}: {m[3]}" if m[3] is not None else box
    if tool == "finish_proofread":
        m = re.fullmatch(r"(\d+) revised", s)
        return t("Agent Result Revised", count=m[1]) if m else s
    return s
