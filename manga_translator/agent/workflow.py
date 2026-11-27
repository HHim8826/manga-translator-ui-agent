"""The AI agent workflow mode: preprocess -> agent -> render, on one MangaTranslator.

1. Preprocess with the existing "Export Original Text" mode: detection, local
   OCR and the refined mask go into ``json/*_translations.json``.
2. The agent works through each chapter folder in page order and writes its
   results back into those JSON files (plus sidecars and an event log).
3. Inpaint review (``inpaint_review``): per finished page, the program inpaints
   block by block and runs the inpainting checks, and a separate review
   conversation fixes what is wrong. The chosen result is saved as the page's
   inpainted image and the JSON mask is updated.
   Optional proofreading (``proofread``) reads the whole chapter's translations once.
   Fonts (``fix_missing_glyphs``, ``choose_fonts``): a region whose font lacks one of its
   characters gets the most similar installed font that has them all; every font change is
   checked by rendering the region with both fonts (see ``fonts.py``).
4. Render with the existing "Import Translation and Render" mode, without the
   TXT import (it would overwrite the agent's JSON). Reviewed pages reuse their
   inpainted image; other pages ignore older inpainted images (the agent may
   have changed the mask).

The editor opens the same JSON files afterwards. A step that could not be done (the inpaint review
of a page, the proofreading: the API's usage limit, most often) is written down and done again by
hand later (``redo.py``, ``run_agent_workflow(redo=True)``).
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import redo as redo_steps
from .client import quota_resets
from .events import EventLog, broadcast, new_run_id
from .font_styles import load_work_styles, save_work_styles, work_fonts_path
from .inpaint_review import InpaintReviewer
from .proofread import ProofPage, Proofreader
from .resume import RunGuard, record_saved
from .revise import mark_pending
from . import json_guard
from .runner import AgentRunner, AgentStopped, language_name
from .settings import AgentSettings
from .store import AgentStore
from .store import page_stem
from .work_glossary import update_work_glossary
from .work_memory import bootstrap, save_chapter, seed_chapter, work_memory_dir
from .work_prompt import load_work_prompt, resolve_path

logger = logging.getLogger("manga_translator.agent")

MODE_FLAGS = ("template", "save_text", "load_text", "translate_json_only", "generate_and_export",
              "colorize_only", "upscale_only", "inpaint_only", "replace_translation", "export_from_local_json",
              "skip_txt_import", "ignore_existing_inpainted", "keep_empty_translations")
RENDERED = ("done", "incomplete")
"""Page statuses that are typeset; the other pages have no agent translation, or the user saved
them in the editor during the run (``edited``: theirs, exported from the editor)."""
HELD = ("inpaint_failed", "inpaint_unknown", "output_kept")
"""Pages a check finished but does not typeset: their inpainted image is not up to their mask
(``_inpaint_changes``), or their output image could not be backed up (``_keep_outputs``)."""


def _image_path(image: Any) -> str:
    if isinstance(image, (str, os.PathLike)):
        return os.fspath(image)
    return getattr(image, "name", "") or ""


def _set_mode(translator, **flags: bool) -> None:
    for name in MODE_FLAGS:
        setattr(translator, name, bool(flags.get(name, False)))


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _overwriting(save_info: Optional[dict]) -> Optional[dict]:
    return None if save_info is None else {**save_info, "overwrite": True}


def _without_upscaling(images_with_configs: List[tuple]) -> List[tuple]:
    """Turns upscaling off in the pages' configs for the preprocess; returns (config.upscale, ratio)
    to put back. The preprocess detected on the upscaled page and wrote its boxes and mask at that
    size with the page's own size, and the agent, the inpaint review and the render work on the page
    as it is: every box, crop and mask was off by the ratio (review 2026-11-28 B10). The render
    (``load_text``) never upscales."""
    changed, seen = [], set()
    for _image, cfg in images_with_configs:
        upscale = getattr(cfg, "upscale", None)
        if upscale is None or id(upscale) in seen or not getattr(upscale, "upscale_ratio", None):
            continue
        seen.add(id(upscale))
        changed.append((upscale, upscale.upscale_ratio))
        try:
            upscale.upscale_ratio = None
        except Exception:  # a frozen config: left as it is
            changed.pop()
    return changed


def _has_output(translator, path: str, save_info: dict) -> bool:
    try:
        target = translator._calculate_output_path(path, save_info)
    except Exception:
        return False
    return bool(target) and os.path.exists(target)


def _rect(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        return all(math.isfinite(float(v)) for v in value)
    except (TypeError, ValueError):
        return False


def editor_box_center(region: dict, editor_layout: bool = True) -> None:
    """Typeset a region in the box set in the editor, as the editor's export does
    (``EditorControllerExportService.apply_white_frame_center``): the center moves to the middle of
    the user's own white frame, else of the saved text box, else of an older file's white frame. A
    region without one (new or changed by the agent) stays centered on its lines.

    The render reads the JSON with ``load_text``, which used only the center of the detected lines:
    a ch34 p6 label the user had moved and spaced by hand came out shifted in the check-again run.

    ``editor_layout``: the page is typeset the way the editor exports it (``skip_font_scaling``,
    which a JSON the editor saved leaves at its default): the center is the middle of the text box.
    A page the pipeline typeset last and the user has not saved since (``skip_font_scaling`` false)
    places its text from the center of the lines itself, at their top or in the middle of the
    bubble, so its box is where the pipeline put it: moving the center there moved the text twice as
    far (ch33 p1: a recap in the margin went 330 px up, off the top of the page). Left as it is."""
    if not editor_layout:
        return
    custom, saved = region.get("white_frame_rect_local"), region.get("render_box_rect_local")
    if region.get("has_custom_white_frame") and _rect(custom):
        box = custom
    elif _rect(saved):
        box = saved
    elif _rect(custom):
        box = custom
    else:
        return
    try:
        cx, cy = (float(v) for v in region["center"][:2])
    except (KeyError, TypeError, ValueError):  # the editor's fallback: the middle of the lines
        points = [p for line in region.get("lines") or [] for p in line]
        if not points:
            return
        xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    left, top, right, bottom = (float(v) for v in box)
    lx, ly = (left + right) / 2, (top + bottom) / 2
    angle = math.radians(float(region.get("angle") or 0.0))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    region["center"] = [cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a]


class PipelineInpaintBackend:
    """The translator's inpainting models and text detector, for the inpaint review."""

    EXTRA_METHODS = ("lama_large",)
    """Offered besides the configured inpainter. Not flux2-klein: on the README's hard page it drew the
    erased labels back from its reference image, or kept whatever the hole was filled with (white, a
    blur), never the dashed lines and screentone lama_large restores, and it took 13-100 s a unit to
    lama's 0.1 s. It is still offered when it is the configured inpainter."""

    def __init__(self, translator, config):
        self.translator = translator
        self.config = config
        self._methods: Optional[List[str]] = None

    def methods(self) -> List[str]:
        """The configured inpainter first, then other models already on disk (never a download)."""
        if self._methods is None:
            from ..config import Inpainter
            from ..inpainting import get_inpainter

            configured = getattr(self.config.inpainter, "inpainter", "")
            configured = getattr(configured, "value", configured) or ""
            names: List[str] = []
            for name in (str(configured), *self.EXTRA_METHODS):
                if not name or name in names or name in ("none", "original"):
                    continue
                try:
                    check = getattr(get_inpainter(Inpainter(name)), "_check_downloaded", None)
                    if check is not None and not check():
                        continue
                except Exception:
                    continue
                names.append(name)
            self._methods = names
        return list(self._methods)

    async def inpaint(self, method: str, image, mask):
        from ..config import Inpainter
        from ..inpainting import dispatch as dispatch_inpainting
        from ..inpainting.ballon_fill import inpaint_regions_per_block

        cfg = self.config.inpainter
        key = Inpainter(method)

        async def block(crop, msk):
            self.translator._check_cancelled()
            return await dispatch_inpainting(key, crop, msk, cfg, cfg.inpainting_size, self.translator.device,
                                             self.translator.verbose)

        result, _count = await inpaint_regions_per_block(image, mask.copy(), block)
        return result

    async def text_map(self, image):
        from ..inpainting.quality_checks import detect_text_map
        return await detect_text_map(image, self.config.detector, self.translator.device)

    def save(self, image_path: str, image) -> str:
        """Writes the reviewed image; raises when it was not written (the translator returns None)."""
        saved = self.translator._save_inpainted_image(image_path, image)
        if not saved:
            raise OSError(f"the inpainted image of {os.path.basename(image_path)} was not saved")
        return saved


async def _proofread(proofreader: Proofreader, pages: List[str], events: EventLog, log) -> dict:
    """The proofreading pass; an API failure is logged and the chapter goes on to render as it is. The
    pages are written down for a redo (``redo.py``) until a proofreading of them ends."""
    stems = [page_stem(p) for p in pages]
    try:
        result = await proofreader.run(pages)
    except (AgentStopped, asyncio.CancelledError):
        raise
    except Exception as exc:
        logger.exception("proofreading failed")
        resets = quota_resets(exc)
        events.emit("error", agent="proofread", error=f"proofreading failed: {type(exc).__name__}: {str(exc)[:300]}",
                    redo=True, **resets)
        _note(redo_steps.failed, proofreader.store, redo_steps.PROOFREAD, stems, str(exc), resets.get("resets_at", ""))
        log(f"[Agent] Proofread failed; it can be done again by hand: start the AI Agent on the chapter and "
            f"choose to redo the steps that were not done"
            + (f" (the API usage limit resets about {resets['resets']})" if resets else ""))
        return {"status": "error", "error": str(exc), **resets}
    if result.get("status") == "incomplete":  # its revisions stay; the rest can be done by hand
        _note(redo_steps.failed, proofreader.store, redo_steps.PROOFREAD, stems,
              f"the proofreading stopped at its step limit ({result.get('steps', 0)} steps) before it finished")
        log("[Agent] Proofread stopped at its step limit before it finished; its revisions stay, and the rest "
            "can be done by hand: start the AI Agent on the chapter and choose to redo the steps that were not done")
    else:
        _note(redo_steps.done, proofreader.store, redo_steps.PROOFREAD, stems)
    totals = proofreader.totals
    log(f"[Agent] Proofread: {result.get('revised', 0)} revised, {result.get('flagged', 0)} flagged, "
        f"{totals['requests']} requests, {totals['prompt_tokens']} prompt tokens ({totals['cached_tokens']} cached)")
    return result


def _note(change, store, *args) -> None:
    """A change to the list of steps left to redo; a failure to write it does not stop the run."""
    try:
        change(store, *args)
    except Exception:
        logger.warning("agent: could not update the steps left to redo", exc_info=True)


def _note_review(store: AgentStore, review: List[dict]) -> None:
    """The pages whose inpaint review failed are written down for a redo, the reviewed ones taken off."""
    for item in review:
        if item.get("status") == "reviewed" or item.get("nothing_to_do"):  # no mask: nothing to redo
            _note(redo_steps.done, store, redo_steps.INPAINT_REVIEW, [item["page"]])
        elif item.get("status") == "error":
            _note(redo_steps.failed, store, redo_steps.INPAINT_REVIEW, [item["page"]], str(item.get("error") or ""),
                  str(item.get("resets_at") or ""))


def _save_work_terms(path: str, memory, pages: List[str], work_dir: str, events: EventLog, log,
                     target_lang: str = "CHT") -> bool:
    """Add the chapter's new names and terms to the work's prompt file; True when it changed.
    A failure is logged and the chapter goes on."""
    chapter = os.path.basename(os.path.dirname(os.path.abspath(work_dir)))
    try:
        # the saved translations, after the proofreading (what the editor and the renderer use)
        translations = [ProofPage.translation(r) for p in pages for r in ProofPage(p).regions.values()]
        result = update_work_glossary(path, memory, translations, chapter, target_lang)
    except Exception as exc:
        logger.exception("saving new terms to the work prompt file failed")
        events.emit("error", agent="memory", error=f"new terms not saved to {os.path.basename(path)}: "
                                                   f"{type(exc).__name__}: {str(exc)[:300]}")
        log(f"[Agent] Work prompt: new terms not saved ({type(exc).__name__}: {exc})")
        return False
    skipped = {reason: len(names) for reason, names in result["skipped"].items()}
    events.emit("work_glossary", agent="memory", file=result["file"], added=result["added"], skipped=skipped)
    added = ", ".join(f"{t['source']} → {t['target']}" for t in result["added"])
    log(f"[Agent] Work prompt: {len(result['added'])} new term(s) saved to {result['file']}"
        + (f": {added}" if added else "") + (f" (left out: {skipped})" if skipped else ""))
    return bool(result["added"])


def _work_styles(path: str, service, log) -> Dict[str, dict]:
    """The work's font styles whose font is installed; the others are left out with a note."""
    styles = load_work_styles(path)
    usable = {name: entry for name, entry in styles.items() if service.catalog.find(entry["font"]) is not None}
    for name in styles.keys() - usable.keys():
        log(f"[Agent] Font style {name}: {styles[name]['font']} is not installed, left out")
    if usable:
        log(f"[Agent] Font styles of the work: " + ", ".join(f"{n} → {e['font']}" for n, e in usable.items()))
    return usable


def _save_work_styles(path: str, memory, work_dir: str, events: EventLog, log) -> None:
    """Keep the chapter's font styles for the next chapters; a failure is logged and the chapter goes on."""
    chapter = os.path.basename(os.path.dirname(os.path.abspath(work_dir)))
    try:
        saved = save_work_styles(path, memory.font_styles, chapter)
    except Exception as exc:
        logger.exception("saving the work's font styles failed")
        events.emit("error", agent="memory", error=f"font styles not saved: {type(exc).__name__}: {str(exc)[:300]}")
        return
    if saved:
        events.emit("font_styles", agent="memory", file=os.path.basename(path),
                    styles={name: memory.font_styles[name]["font"] for name in saved})
        log(f"[Agent] Font styles saved for the work: {', '.join(saved)}")


def _work_memory(settings: AgentSettings, prompt_path: str, log) -> Tuple[str, str]:
    """The work's memory folder and its prompt file's name (``work_memory.py``); empty when the
    chapters do not share one."""
    if settings.memory_dir:
        log(f"[Agent] Memory: {settings.memory_dir} (every chapter)")
        return "", ""
    if not settings.share_memory:
        log("[Agent] Memory: each chapter its own")
        return "", ""
    resolved = resolve_path(prompt_path) if prompt_path else None
    if not resolved:
        log("[Agent] Memory: each chapter its own (a work prompt file lets the chapters share it)")
        return "", ""
    folder = work_memory_dir(resolved)
    log(f"[Agent] Memory shared by the work's chapters: {folder}")
    return folder, os.path.basename(resolved)


def _seed_memory(folder: str, prompt_name: str, store: AgentStore, log) -> Optional[dict]:
    """A chapter's first run starts with what the earlier chapters learned; a failure is logged."""
    chapter = os.path.basename(os.path.dirname(os.path.abspath(store.work_dir)))
    try:
        earlier = bootstrap(folder, store.work_dir, prompt_name)
        if earlier:
            log(f"[Agent] Work memory put together from {len(earlier)} earlier chapter(s): {', '.join(earlier)}")
        seeded = seed_chapter(store.memory(), folder, chapter, os.path.dirname(os.path.abspath(store.work_dir)))
    except Exception as exc:
        logger.exception("reading the work's memory failed")
        log(f"[Agent] {chapter}: the work's memory was not read ({type(exc).__name__}: {exc})")
        return None
    if not seeded:
        return None
    log(f"[Agent] {chapter}: starts with the work's memory: {seeded['glossary']} term(s), "
        f"{seeded['characters']} character(s), story summary of {seeded['story']} characters")
    return {**seeded, "from": earlier}


def _earlier(work_memory: str, store: AgentStore, prompt_name: str):
    """The work's earlier chapters for search_past_translations, or None without the work."""
    if not (work_memory or prompt_name):
        return None
    from .work_memory import EarlierTranslations

    return EarlierTranslations(work_memory, store.work_dir, prompt_name)


async def _synopses(client, settings: AgentSettings, target_lang: str, folder: str, work_dir: str,
                    prompt_name: str, log, should_stop: Callable[[], bool]) -> Tuple[str, Dict[str, Any]]:
    """The synopses of the work's earlier chapters for the prompts (``synopses.py``), the missing ones
    written first: (text, what was written for run_start). Only with the work's memory (share_memory):
    they are kept in its folder. A failure leaves the chapter without them."""
    if not folder:
        return "", {}
    from . import synopses

    try:
        return await synopses.update(client, settings, target_lang, folder, work_dir, prompt_name, log,
                                     should_stop)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("agent: the chapter synopses could not be written", exc_info=True)
        log(f"[Agent] Chapter synopses: not written ({type(exc).__name__}: {str(exc)[:160]}); the chapter "
            "goes on without them")
        return "", {}


def _save_memory(folder: str, memory, work_dir: str, events: EventLog, log) -> None:
    """Merge the chapter's memory into the work's; a failure is logged and the chapter goes on."""
    chapter = os.path.basename(os.path.dirname(os.path.abspath(work_dir)))
    try:
        changes = save_chapter(memory, folder, chapter, os.path.dirname(os.path.abspath(work_dir)))
    except Exception as exc:
        logger.exception("saving the work's memory failed")
        events.emit("error", agent="memory", error=f"the work's memory was not saved: {type(exc).__name__}: "
                                                   f"{str(exc)[:300]}")
        return
    events.emit("work_memory", agent="memory", action="saved", folder=folder, **changes)
    log(f"[Agent] Work memory: {changes['glossary_added']} new term(s), {changes['glossary_updated']} changed, "
        f"{changes['characters_added']} new character(s)")


async def _check_fonts(service, pages: List[str], store: AgentStore, events: EventLog, settings: AgentSettings,
                       log) -> None:
    """The font check of each finished page; a failure leaves that page's fonts as they are."""
    changed = borrowed = unfixed = 0
    for path in pages:
        try:
            result = await service.fix_page(path, store, events, fix_missing=settings.fix_missing_glyphs)
        except (AgentStopped, asyncio.CancelledError):
            raise
        except Exception as exc:
            logger.exception("font check failed")
            events.emit("error", page=page_stem(path), agent="fonts",
                        error=f"font check failed: {type(exc).__name__}: {str(exc)[:300]}")
            continue
        changed += result.get("changed", 0)
        borrowed += result.get("borrowed", 0)
        unfixed += result.get("unfixed", 0)
    log(f"[Agent] Fonts: {changed} region(s) back to the user's font, {borrowed} draw missing characters from "
        f"a similar font, {unfixed} left with characters no font has")


def line_break_rules(config) -> str:
    """The AI line break rules (dict/system_prompt_line_break.yaml, the classic translators' rules)
    when the render setting disable_auto_wrap ("AI line breaks") is on; empty when it is off."""
    if not getattr(getattr(config, "render", None), "disable_auto_wrap", False):
        return ""
    from .runner import LINE_BREAK_RULES

    try:
        from ..translators.prompt_loader import load_line_break_prompt
        from ..utils.generic import BASE_PATH

        data = load_line_break_prompt(os.path.join(BASE_PATH, "dict")) or {}
        rules = str(data.get("line_break_prompt") or "").strip()
    except Exception:
        logger.warning("agent: could not read the line break rules", exc_info=True)
        rules = ""
    return rules or LINE_BREAK_RULES


def _render_backend(translator, config, font_service):
    """The renderer for the size check at mark_page_done (``render_check.py``): the font service's,
    or one of its own when the font features are off; None when the pipeline cannot load a page from
    memory."""
    backend = getattr(font_service, "backend", None)
    if backend is not None:
        return backend
    if not (hasattr(translator, "_load_text_and_regions_from_file")
            and hasattr(translator, "set_preloaded_load_text_payload")):
        return None
    from .fonts import PipelineFontBackend

    return PipelineFontBackend(translator, config)


async def _font_service(translator, config, settings: AgentSettings, target_lang: str, log):
    """Installed fonts and the renderer for the layout check; None when the pipeline has no renderer
    (or the fonts cannot be read), and the agents go on without the font features."""
    if not (settings.choose_fonts or settings.fix_missing_glyphs):
        return None
    if not hasattr(translator, "_load_text_and_regions_from_file"):
        return None
    from .fonts import build_font_service

    try:
        service = await build_font_service(translator, config, target_lang)
        if settings.choose_fonts:
            await asyncio.to_thread(service.prepare_sheet)
    except (AgentStopped, asyncio.CancelledError):
        raise
    except Exception as exc:
        logger.exception("font catalog failed")
        log(f"[Agent] Fonts: not available ({type(exc).__name__}: {str(exc)[:200]})")
        return None
    usable = sum(1 for face in service.catalog.faces if service.catalog.usable(face))
    log(f"[Agent] Fonts: {usable} installed face(s), {len(service.choices)} on the agent's font sheet, "
        f"default {service.default_font}")
    return service


class _RenderEvents:
    """``render_start``/``render_done`` in each chapter's run log, so the Agents page can show
    every page's result as soon as it is saved (and a replayed log finds it too)."""

    def __init__(self, chapters: Dict[str, Tuple[str, str]]):
        self.chapters = chapters  # work dir -> (run log path, run id)
        self.logs: Dict[str, EventLog] = {}
        self.pending: Dict[str, str] = {}
        """Pages being typeset that have no ``render_done`` yet (key -> image path)."""

    def _log(self, image_path: str) -> Optional[EventLog]:
        from ..utils.path_manager import get_work_dir

        work_dir = get_work_dir(image_path)
        if work_dir not in self.chapters:
            return None
        if work_dir not in self.logs:
            path, run_id = self.chapters[work_dir]
            self.logs[work_dir] = EventLog(path, run_id)
        return self.logs[work_dir]

    def start(self, paths: Sequence[str], left_out: Optional[Dict[str, str]] = None) -> None:
        """``left_out``: pages that are not typeset → their agent status (``error``, ``stopped``,
        ``not_reached`` ...), listed in their chapter's ``render_start``."""
        left_out = left_out or {}
        rendered = {_key(p) for p in paths}
        self.pending = {_key(p): p for p in paths}
        for work_dir, pages in group_by_chapter(list(paths) + list(left_out)):
            log = self._log(pages[0])
            if log is not None:
                skipped = {page_stem(p): left_out[p]
                           for p in pages if _key(p) not in rendered and p in left_out}
                log.emit("render_start", agent="render", pages=sum(1 for p in pages if _key(p) in rendered),
                         **({"skipped": skipped} if skipped else {}))

    def saved(self, ctx) -> None:  # the translator's on_result_saved hook
        from ..utils.path_manager import find_inpainted_path

        image = _image_path(getattr(ctx, "image_name", "") or "")
        if not image:
            return
        error = getattr(ctx, "translation_error", None)
        if error:  # the pipeline could not typeset the page (out of memory, say) and saved it untranslated
            self._end(image, "failed", error=str(error), output=getattr(ctx, "output_path", None))
            return
        self.pending.pop(_key(image), None)
        log = self._log(image)
        if log is not None:
            log.emit("render_done", page=page_stem(image), agent="render",
                     image=image, output=getattr(ctx, "output_path", None), inpainted=find_inpainted_path(image),
                     result="saved")

    def _end(self, image: str, result: str, **fields) -> None:
        self.pending.pop(_key(image), None)
        log = self._log(image)
        if log is not None:
            log.emit("render_done", page=page_stem(image), agent="render",
                     image=image, result=result, **fields)

    def finish(self, results: Sequence) -> None:
        """The pages typeset without a saved result: skipped (an output file is already there and
        overwriting is off ...) or failed (the render or the save went wrong)."""
        by_key = {}
        for ctx in results:
            name = getattr(ctx, "image_name", None)
            if isinstance(name, str) and name:
                by_key[_key(_image_path(name))] = ctx
        for key, image in list(self.pending.items()):
            ctx = by_key.get(key)
            if ctx is not None and getattr(ctx, "skipped", False):
                self._end(image, "skipped", reason=str(getattr(ctx, "skip_reason", "") or ""),
                          message=str(getattr(ctx, "skip_message", "") or ""), output=getattr(ctx, "output_path", None))
            else:
                error = getattr(ctx, "translation_error", None) if ctx is not None else None
                self._end(image, "failed", error=str(error or "the result was not saved"))

    def stopped(self) -> None:
        for image in list(self.pending.values()):
            self._end(image, "stopped")
        for log in self.logs.values():
            log.emit("stopped", agent="workflow", stage="render")

    def close(self) -> None:
        for log in self.logs.values():
            log.close()
        self.logs.clear()


async def _review_queue(reviewer: InpaintReviewer, queue: "asyncio.Queue") -> List[dict]:
    """Review pages as they arrive until ``None``."""
    outcomes: List[dict] = []
    while True:
        path = await queue.get()
        if path is None:
            return outcomes
        outcomes.extend(await reviewer.run([path]))


async def _inpaint_changes(pages: List[str], guard: RunGuard, backend, store: AgentStore, events: EventLog, log,
                           should_stop: Callable[[], bool]) -> Tuple[set, Dict[str, str]]:
    """A check: each finished page's inpainted image brought up to its new mask, the rest kept
    (``revise.inpaint_changes``). Returns (the pages whose render reuses their inpainted image, the
    pages held back → why).

    A held page keeps its inpainted image and its output image: the render does not inpaint it
    again over the user's repairs. ``inpaint_failed``: the patch failed; ``inpaint_unknown``: a mask
    is missing and the boxes changed. Its pending mark (``revise.mark_pending``) stays, so the next
    check patches it from the mask its image still matches."""
    from ..utils.path_manager import find_json_path
    from .revise import clear_pending, inpaint_changes, pending_before

    reuse, held = set(), {}
    for path in pages:
        if should_stop():
            raise asyncio.CancelledError("Agent workflow stopped by user")
        if json_guard.taken(find_json_path(path)):  # saved in the editor: the user's, not rendered
            continue
        page = page_stem(path)
        # an earlier check that stopped before this step left the mask its image still matches
        before = pending_before(store, page, path) or (guard.entries.get(_key(path)) or {}).get("json_copy") or ""
        events.emit("inpaint_changes", page=page, agent="inpaint", status="working")
        try:
            result = await inpaint_changes(path, before, backend, backend.save)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"inpainting the changes of {page} failed", exc_info=True)
            log(f"[Agent] {page}: inpainting the changed areas failed ({type(exc).__name__}: {exc}); the page "
                f"keeps its inpainted image and is not rendered again. Check it again to retry")
            events.emit("inpaint_changes", page=page, agent="inpaint", status="error", image=path,
                        error=f"{type(exc).__name__}: {str(exc)[:200]}")
            held[_key(path)] = "inpaint_failed"
            continue
        events.emit("inpaint_changes", page=page, agent="inpaint", image=path, **result)
        if result["status"] == "no_mask":
            log(f"[Agent] {page}: the mask from before or after the check is missing and the text boxes "
                f"changed, so what to inpaint cannot be told; the page keeps its inpainted image and is not "
                f"rendered again")
            held[_key(path)] = "inpaint_unknown"
            continue
        clear_pending(store, page)
        if result["status"] != "no_inpainted":
            reuse.add(_key(path))
        if result["status"] == "patched":
            log(f"[Agent] {page}: inpainted again where text is erased now ({result['added']} px), original "
                f"back where it is kept now ({result['restored']} px); the rest of the image stays")
    return reuse, held


def _keep_outputs(translator, items: List[tuple], save_info: Optional[dict], log) -> Dict[str, str]:
    """A check renders the pages again over their output images (the user's exports from the editor):
    a copy of each goes to ``agent/backups`` first. Returns the pages whose output image could not be
    copied (key -> image path): they are not rendered, so the image is not overwritten."""
    import shutil
    import time

    from ..utils.path_manager import get_work_dir

    stamp = time.strftime("%Y%m%d-%H%M%S")
    kept, unsaved = 0, {}
    for image, _cfg in items:
        path = _image_path(image)
        try:
            target = translator._calculate_output_path(path, save_info or {})
        except Exception:
            continue
        if not target or not os.path.exists(target):
            continue
        folder = os.path.join(AgentStore(get_work_dir(path)).agent_dir, "backups")
        base, ext = os.path.splitext(os.path.basename(target))
        try:
            os.makedirs(folder, exist_ok=True)
            shutil.copy2(target, os.path.join(folder, f"{base}.output.{stamp}{ext}"))
            kept += 1
        except OSError as exc:
            logger.warning(f"agent: could not copy {target}", exc_info=True)
            log(f"[Agent] {os.path.basename(path)}: the output image could not be copied to agent/backups "
                f"({type(exc).__name__}: {exc}); it is not rendered again, so the image stays as it is")
            unsaved[_key(path)] = path
    if kept:
        log(f"[Agent] {kept} output image(s) from before the check copied to agent/backups")
    return unsaved


def _left_out_context(path: str, status: str, error: str = ""):
    """A result for a page the render left out, so the batch summary names it: a failure with its
    reason when the agent failed on it, else a skipped page (stopped, not reached)."""
    from ..utils import Context

    ctx = Context()
    ctx.image_name = path
    if status in HELD:
        ctx.success = False
        why = {"inpaint_failed": "inpainting the changed areas failed",
               "inpaint_unknown": "the page has no mask to compare and its text boxes changed",
               "output_kept": "its output image could not be backed up"}[status]
        ctx.translation_error = ctx.error = (
            f"AI Agent: {why}; the page keeps its inpainted and output images and was not rendered again "
            f"(check it again to retry)")
        return ctx
    if status in ("redo_failed", "redo_none", "redo_yours"):
        ctx.skipped = True
        ctx.skip_reason = f"agent_{status}"
        ctx.skip_message = {
            "redo_failed": "AI Agent: the step to redo failed again; the page keeps its output and its step "
                           "stays to redo",
            "redo_none": "AI Agent: nothing was redone on this page; it keeps its output",
            "redo_yours": "AI Agent: you changed this page after the run; its inpaint review is not done again "
                          "and it keeps its output"}[status]
        return ctx
    if status in ("error", "no_json"):
        ctx.success = False
        ctx.translation_error = ctx.error = (f"AI Agent: {error}" if error else
                                             "AI Agent: the page has no JSON after the agent" if status == "no_json"
                                             else "AI Agent: the agent failed on this page")
        return ctx
    ctx.skipped = True
    ctx.skip_reason = f"agent_{status}"
    ctx.skip_message = f"AI Agent: no translation ({status}), not rendered"
    return ctx


def group_by_chapter(paths: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """(work dir, pages) per folder, folders in first-seen order, pages in input order."""
    from ..utils.path_manager import get_work_dir

    groups: Dict[str, List[str]] = {}
    for path in paths:
        groups.setdefault(get_work_dir(path), []).append(path)
    return list(groups.items())


def _report(progress, stage: str, done: int, total: int) -> None:
    if progress is not None:
        try:
            progress(stage, done, total)
        except Exception:
            logger.exception("agent progress callback failed")


class _BatchProgress:
    """The translator's ``batch:<start>:<end>:<total>`` reports during the preprocess and the render,
    as progress of that stage. ``offset`` adds the pages of an earlier render batch."""

    def __init__(self, progress, total: int):
        self.progress = progress
        self.total = total
        self.stage: Optional[str] = None
        self.offset = 0

    async def __call__(self, state, finished=False) -> None:
        if self.stage is None or not isinstance(state, str) or not state.startswith("batch:"):
            return
        parts = state.split(":")
        try:
            done = int(parts[2])
        except (IndexError, ValueError):
            return
        _report(self.progress, self.stage, min(self.offset + done, self.total), self.total)


async def _redo_chapter(pages: List[str], store: AgentStore, events: EventLog, client, settings: AgentSettings,
                        target_lang: str, work_prompt, font_service, line_breaks: bool,
                        should_stop: Callable[[], bool], backend, guard: Optional[RunGuard], log,
                        run_info: Dict[str, Any], work_memory: str = "",
                        prompt_name: str = "") -> Tuple[List[dict], set]:
    """The steps an earlier run could not do on a chapter's pages (``redo.py``): the inpaint review of
    each page, then the proofreading, then the font check of the proofread pages. No translation:
    the pages keep their text, and the proofreader leaves the boxes the user changed in the editor.
    Returns (each page's outcome for the render, the pages whose render reuses their inpainted image:
    every page a step was done on, since a redo does not change the masks the image was made for, and
    the review saved its own). A page is rendered again when a step was done on it; one whose steps
    failed again (``redo_failed``) keeps its output and stays on the list; one whose only step was
    the review of a page the user changed (its JSON, or only its inpainted image: their repairs) keeps
    its output (``redo_yours``), and so does one with nothing done (``redo_none``)."""
    from .resume import previous_states
    from .revise import load_page

    pending = redo_steps.read(store)
    stems = {p: page_stem(p) for p in pages}
    states = {_key(s.path): s for s in previous_states(pages)}
    synopsis_text, written = "", {}
    if settings.proofread and any(stems[p] in pending.get(redo_steps.PROOFREAD, {}) for p in pages):
        synopsis_text, written = await _synopses(client, settings, target_lang, work_memory, store.work_dir,
                                                 prompt_name, log, should_stop)
    if written:
        run_info = {**run_info, "synopses": written}
    events.emit("run_start", pages=[os.path.basename(p) for p in pages], stems=[stems[p] for p in pages],
                model=getattr(client, "model_label", ""),
                target_lang=target_lang, redo=redo_steps.summary(
                    {step: {k: v for k, v in pending.get(step, {}).items() if k in stems.values()}
                     for step in redo_steps.STEPS}), **run_info)
    worked: Dict[str, bool] = {}
    """Page -> a step was done on it (rendered again); False: its steps failed again."""
    yours = set()
    """Pages whose inpaint review is not done again: the user changed them."""
    requests = 0
    ended = ""
    try:
        to_review = []
        for path in pages:
            if stems[path] not in pending.get(redo_steps.INPAINT_REVIEW, {}):
                continue
            state = states.get(_key(path))
            if getattr(state, "edited", False) or getattr(state, "inpaint_edited", False):
                # its inpainted image may hold the user's repairs: the review would paint over them
                log(f"[Agent] {stems[path]}: " + ("changed in the editor" if getattr(state, "edited", False) else
                                                  "its inpainted image changed") +
                    " after the run; its inpainting is yours, the inpaint review is not done again")
                _note(redo_steps.done, store, redo_steps.INPAINT_REVIEW, [stems[path]])
                yours.add(_key(path))
                continue
            to_review.append(path)
        if to_review:
            log(f"[Agent] Redo: inpaint review of {len(to_review)} page(s)")
            reviewer = InpaintReviewer(settings, client, store, events, target_lang, backend, should_stop,
                                       save_image=getattr(backend, "save", None))
            review = await reviewer.run(to_review)
            _note_review(store, review)
            requests += reviewer.totals["requests"]
            for item in review:
                ok = item.get("status") == "reviewed"
                worked[_key(item["image"])] = worked.get(_key(item["image"]), False) or ok
        to_proofread = [p for p in pages if stems[p] in pending.get(redo_steps.PROOFREAD, {})]
        if to_proofread:
            log(f"[Agent] Redo: proofreading {len(to_proofread)} page(s); the boxes you changed are "
                f"{'left alone' if settings.revise_lock_user_edits else 'not locked'}")
            locked: Dict[str, set] = {}
            if settings.revise_lock_user_edits:
                for path in to_proofread:
                    try:
                        doc = load_page(path, store)
                    except Exception:
                        logger.warning("agent: cannot tell which boxes of %s the user changed", path, exc_info=True)
                        continue
                    locked[stems[path]] = {rid for rid, r in doc.regions.items() if r.locked and not r.was_skipped}
            proofreader = Proofreader(settings, client, store, events, target_lang, store.memory(), work_prompt,
                                      should_stop, line_breaks=line_breaks, locked=locked,
                                      synopses=synopsis_text, earlier=_earlier(work_memory, store, prompt_name))
            result = await _proofread(proofreader, to_proofread, events, log)
            requests += proofreader.totals["requests"]
            ok = result.get("status") != "error"
            for path in to_proofread:
                worked[_key(path)] = worked.get(_key(path), False) or ok
            if ok and font_service is not None:  # the proofreading may change the text
                await _check_fonts(font_service, to_proofread, store, events, settings, log)
    except (AgentStopped, asyncio.CancelledError):
        ended = "stopped"
        raise
    finally:
        outcomes = []
        for path in pages:
            side = store.load_sidecar(stems[path]) or {}
            status = str(side.get("status") or "done")
            if not worked.get(_key(path)):
                status = "redo_failed" if _key(path) in worked else "redo_yours" if _key(path) in yours else \
                    "redo_none"
            elif guard is not None:
                guard.page_finished(path)
            flags = len(side.get("review_flags") or [])
            outcomes.append({"page": stems[path], "image": path, "status": status, "flags": flags})
            if ended != "stopped":
                events.emit("page_done", page=stems[path], status=status, flags=flags, image=path)
        events.emit("run_end", pages=len(outcomes), total=len(pages), reason=ended,
                    done=sum(1 for o in outcomes if o["status"] in RENDERED), flagged=0,
                    totals={"requests": requests}, redo=True)
    return outcomes, {key for key, ok in worked.items() if ok}


async def run_agent_workflow(
    translator,
    images_with_configs: List[tuple],
    save_info: Optional[dict] = None,
    *,
    agent_settings: Union[AgentSettings, dict, None] = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], Any] = logger.info,
    progress: Optional[Callable[[str, int, int], Any]] = None,
    client=None,
    preprocess: bool = True,
    render: bool = True,
    inpaint_backend=None,
    fonts=None,
    revise: bool = False,
    redo: bool = False,
) -> list:
    """``revise`` checks again pages an earlier run finished (``revise.py``): no preprocess, the
    user's changes stay, only what changed is inpainted again. ``redo`` does only the steps an
    earlier run could not do on the pages (``redo.py``), then renders them again."""
    settings = agent_settings if isinstance(agent_settings, AgentSettings) else AgentSettings(**(agent_settings or {}))
    if redo:
        revise = False
    if revise or redo:
        preprocess = False
    if not images_with_configs:
        return []
    paths = [_image_path(image) for image, _ in images_with_configs]
    render_info = save_info
    if save_info is not None and not save_info.get("overwrite", True):
        if revise or redo:
            # their output images are copied to agent/backups first (_keep_outputs) and the run is
            # there to render them again: with overwrite off the render skipped every page
            render_info = _overwriting(save_info)
        else:
            # as a translation with overwrite off: a page that has its output image is left out, here
            # before the API works on it (the render would skip it afterwards) (review 2026-11-28 B11)
            kept = {_key(path) for path in paths if _has_output(translator, path, save_info)}
            if kept:
                log(f"[Agent] Overwrite is off: {len(kept)} page(s) already have an output image and are left "
                    "as they are: " + ", ".join(os.path.basename(p) for p in paths if _key(p) in kept))
                images_with_configs = [item for item in images_with_configs
                                       if _key(_image_path(item[0])) not in kept]
                if not images_with_configs:
                    return []
                paths = [_image_path(image) for image, _ in images_with_configs]
    config = images_with_configs[0][1]
    saved = {name: getattr(translator, name, False) for name in MODE_FLAGS}
    target_lang = getattr(getattr(config, "translator", None), "target_lang", "") or "CHT"
    font_task: Optional[asyncio.Future] = None
    own_client = client is None
    """The workflow builds the API client, so it closes it at the end."""
    guard: Optional[RunGuard] = None
    """Copies of the pages from before the run: the pages it does not finish get them back."""
    preprocess_failed: Dict[str, str] = {}
    chapter_logs: Dict[str, Tuple[str, str]] = {}
    """Work dir -> (run log path, run id) of the chapters started so far."""
    # progress(stage, done, total): ``preprocess`` and ``render`` count the translator's pages,
    # ``translate`` the pages the agent finished, ``proofread`` starts a chapter's proofreading
    batch_progress = _BatchProgress(progress, len(paths))
    hooks = getattr(translator, "_progress_hooks", None)
    if progress is not None and isinstance(hooks, list):
        hooks.append(batch_progress)
    try:
        # until a chapter's run_start the AI Agent page hears of the run only from these
        broadcast("workflow_stage", stage="start", pages=len(paths))
        if fonts is None:  # scanning the installed fonts takes seconds: do it while the pages are preprocessed
            font_task = asyncio.ensure_future(_font_service(translator, config, settings, target_lang, log))
        if own_client:  # before the preprocess: a wrong API setting should not wait for it
            from .client import build_client, check_api
            client = build_client(config, settings)
            log(f"[Agent] API check: model {getattr(client, 'model_label', '')}")
            broadcast("workflow_stage", stage="api_check", model=getattr(client, "model_label", ""))
            await check_api(client)
        if preprocess:
            # the preprocess writes every page's JSON again: copies first, and the pages the run does
            # not finish get theirs back (resume.RunGuard)
            guard = RunGuard.begin(paths)
            if guard.users_copies:
                log(f"[Agent] {len(guard.users_copies)} JSON file(s) with your own changes or translations, "
                    f"copied before the preprocess replaces them: "
                    f"{', '.join(os.path.basename(c) for c in guard.users_copies)}")
            log("[Agent] 1/4 Preprocess: detection, local OCR and masks")
            _set_mode(translator, template=True, save_text=True)
            batch_progress.stage = "preprocess"
            _report(progress, "preprocess", 0, len(paths))
            broadcast("workflow_stage", stage="preprocess", pages=len(paths))
            upscaled = _without_upscaling(images_with_configs)
            if upscaled:
                log(f"[Agent] Upscaling (x{upscaled[0][1]}) is off for the AI Agent: it works on the pages as "
                    "they are")
            try:
                # every page's JSON and TXT are written again (copies first): with overwrite off an older
                # TXT made the page skipped, and taken as failed
                preprocessed = await translator.translate_batch(images_with_configs,
                                                                save_info=_overwriting(save_info))
            finally:
                for upscale, ratio in upscaled:
                    upscale.upscale_ratio = ratio
            batch_progress.stage = None
            preprocess_failed = guard.preprocess_failures(preprocessed)
            from ..utils.path_manager import find_json_path as _find_json

            for path in paths:  # from here on, a save in the editor makes the page the user's
                json_guard.note(_find_json(path))
            if preprocess_failed:
                log("[Agent] Preprocess failed, the agent leaves these pages as they were: " + "; ".join(
                    f"{os.path.basename(p)} ({why})" for p, why in preprocess_failed.items()))
        elif revise:
            # the pages change in place: copies first (JSON, sidecar, inpainted image), and the pages
            # the check does not finish get theirs back. The AI Agent page shows this stage (with
            # its stop button) until the first chapter starts.
            broadcast("workflow_stage", stage="revise", pages=len(paths))
            guard = RunGuard.begin(paths, revise=True)
            from ..utils.path_manager import find_json_path as _find_json

            for path in paths:  # from here on, a save in the editor makes the page the user's
                json_guard.note(_find_json(path))
            log(f"[Agent] Check again: {len(paths)} page(s) keep their translations and your changes; copies of "
                f"them are in agent/backups. Your edited boxes are "
                f"{'locked' if settings.revise_lock_user_edits else 'not locked'}.")
        elif redo:
            # like a check: the pages change in place, copies first
            broadcast("workflow_stage", stage="redo", pages=len(paths))
            guard = RunGuard.begin(paths, revise=True)
            from ..utils.path_manager import find_json_path as _find_json

            for path in paths:
                json_guard.note(_find_json(path))
            log(f"[Agent] Redo: the steps an earlier run could not do on {len(paths)} page(s); the pages keep "
                f"their translations and your changes (copies in agent/backups)")
        if should_stop():
            raise asyncio.CancelledError("Agent workflow stopped by user")

        log(f"[Agent] 2/4 Agent: {len(paths)} page(s), model {getattr(client, 'model_label', '')}, "
            f"reasoning effort {settings.reasoning_effort or 'default'} "
            f"(recall and inpaint review: {settings.review_reasoning_effort or 'default'})")
        work_prompt = None
        prompt_path = getattr(getattr(config, "translator", None), "high_quality_prompt_path", None) or ""
        if settings.use_work_prompt:
            work_prompt = load_work_prompt(prompt_path, language_name(target_lang))
            if work_prompt is not None:
                log(f"[Agent] Work prompt: {work_prompt.name}, {len(work_prompt.terms)} glossary term(s)")
        work_memory, prompt_name = _work_memory(settings, prompt_path, log)
        font_service = fonts if font_task is None else await font_task
        render_backend = _render_backend(translator, config, font_service)
        breaks = line_break_rules(config)
        if breaks:
            log("[Agent] AI line breaks: on (translations get [BR] line breaks by the source line count)")
        outcomes: List[dict] = []
        translated = [0]
        _report(progress, "translate", 0, len(paths))
        reviewed: set = set()
        held: Dict[str, str] = {}
        """Pages a check holds back from the render (_inpaint_changes) -> why."""
        batch_id = f"b{uuid.uuid4().hex[:12]}"  # the chapters of this run, for the AI Agent page
        for work_dir, pages in group_by_chapter(paths):
            store = AgentStore(work_dir, settings.memory_dir)
            seeded = _seed_memory(work_memory, prompt_name, store, log) if work_memory else None
            run_id = new_run_id()
            taken = {run for _, run in chapter_logs.values()}
            base, number = run_id, 1
            while run_id in taken:  # two chapters started in the same second
                number += 1
                run_id = f"{base}~{number}"
            events = EventLog(store.run_log_path(run_id), run_id)
            chapter_logs[work_dir] = (events.path, run_id)
            if preprocess:  # translated again: what was left to do on these pages goes with their old text
                _note(redo_steps.forget, store, [page_stem(p) for p in pages])
            log(f"[Agent] {os.path.basename(os.path.dirname(work_dir))}: event log {events.path}")
            if hasattr(client, "session_id"):  # one session per chapter run on the API side
                client.session_id = f"manga-translator-{run_id}-{uuid.uuid4().hex[:8]}"
            try:
                if redo:
                    try:
                        chapter, redone = await _redo_chapter(
                            pages, store, events, client, settings, target_lang, work_prompt, font_service,
                            bool(breaks), should_stop, inpaint_backend or PipelineInpaintBackend(translator, config),
                            guard, log, {"chapter": os.path.basename(os.path.dirname(work_dir)), "batch": batch_id},
                            work_memory=work_memory, prompt_name=prompt_name)
                    except AgentStopped:  # the stop button, as in a run (review 2026-11-28 B9)
                        raise asyncio.CancelledError("Agent workflow stopped by user")
                    reviewed |= redone
                else:
                    reviewer = None
                    # a check keeps the inpainted image (the user's repairs are in it): only what changed is
                    # inpainted again, after the agent (_inpaint_changes)
                    if settings.inpaint_review and not revise:
                        backend = inpaint_backend or PipelineInpaintBackend(translator, config)
                        reviewer = InpaintReviewer(settings, client, store, events, target_lang, backend, should_stop,
                                                   save_image=getattr(backend, "save", None))
                    queue: asyncio.Queue = asyncio.Queue()
                    review_task: Optional[asyncio.Future] = None
                    if reviewer is not None and settings.review_while_translating:
                        log("[Agent] 3/4 Inpaint review: each page as soon as the agent finishes it")
                        review_task = asyncio.ensure_future(_review_queue(reviewer, queue))

                    def page_done(outcome: dict) -> None:
                        if preprocess and outcome.get("status") in ("done", "incomplete"):
                            # Same as "Translate JSON Only": the original TXT this run's preprocess
                            # exported would otherwise be imported over the agent's JSON later. Now,
                            # not at the end of the chapter: a stop keeps the finished pages (and
                            # their TXT). A check has no preprocess: a TXT there is the user's.
                            translator._delete_original_txt_after_json_translation(outcome["image"])
                        if guard is not None:
                            guard.page_finished(outcome["image"])
                            if revise and outcome.get("status") in RENDERED:
                                # its JSON keeps this check's mask from now on: until the inpainted image
                                # is brought up to it, the mask it matches is kept (a stop or a failure)
                                entry = guard.entries.get(_key(outcome["image"])) or {}
                                try:
                                    mark_pending(store, outcome["page"], outcome["image"], entry.get("json_copy") or "")
                                except Exception:
                                    logger.warning("agent: could not mark the inpainting as pending", exc_info=True)
                        translated[0] += 1
                        _report(progress, "translate", translated[0], len(paths))
                        if review_task is not None and outcome.get("status") in ("done", "incomplete"):
                            queue.put_nowait(outcome["image"])

                    # a stop meanwhile leaves the rest unwritten; the runner then stops as at any page
                    synopsis_text, written = await _synopses(client, settings, target_lang, work_memory, work_dir,
                                                             prompt_name, log, should_stop)
                    use_styles = work_prompt is not None and settings.choose_fonts and font_service is not None
                    styles_path = work_fonts_path(work_prompt.path) if use_styles else ""
                    runner = AgentRunner(settings, client, store, events, pages, target_lang, should_stop,
                                         on_page_done=page_done, work_prompt=work_prompt,
                                         fonts=font_service if settings.choose_fonts else None, line_breaks=breaks,
                                         font_styles=_work_styles(styles_path, font_service, log) if styles_path else None,
                                         run_info={"chapter": os.path.basename(os.path.dirname(work_dir)), "batch": batch_id,
                                                   **({"work_memory": seeded} if seeded else {}),
                                                   **({"synopses": written} if written else {})},
                                         preprocess_failed=preprocess_failed, revise=revise, revise_fonts=font_service,
                                         default_font=getattr(getattr(config, "render", None), "font_family", "") or "",
                                         render_backend=render_backend, work_memory=work_memory,
                                         prompt_name=prompt_name, synopses=synopsis_text)
                    review: List[dict] = []
                    proof_task: Optional[asyncio.Future] = None
                    try:
                        try:
                            chapter = await runner.run()
                            finished = [o["image"] for o in chapter if o.get("status") in ("done", "incomplete")]
                            if settings.proofread and finished:
                                # text only, so it runs while the review is still busy with the last pages
                                log(f"[Agent] Proofread: {len(finished)} page(s)")
                                _report(progress, "proofread", translated[0], len(paths))
                                proofreader = Proofreader(settings, client, store, events, target_lang, runner.memory,
                                                          work_prompt, should_stop, line_breaks=bool(breaks),
                                                          synopses=synopsis_text,
                                                          earlier=runner.session.earlier_chapters)
                                proof_task = asyncio.ensure_future(_proofread(proofreader, finished, events, log))
                            if review_task is not None:
                                queue.put_nowait(None)
                                review = await review_task
                            elif reviewer is not None and finished:
                                log(f"[Agent] 3/4 Inpaint review: {len(finished)} page(s)")
                                review = await reviewer.run(finished)
                            if proof_task is not None:
                                await proof_task
                            if font_service is not None and finished:
                                # after the proofreading: it may change the text
                                await _check_fonts(font_service, finished, store, events, settings, log)
                            if styles_path and finished:
                                _save_work_styles(styles_path, runner.memory, work_dir, events, log)
                            if settings.update_work_glossary and work_prompt is not None and work_prompt.path \
                                    and finished and _save_work_terms(work_prompt.path, runner.memory, finished,
                                                                      work_dir, events, log, target_lang):
                                # the next chapter reads the file again, with the new terms
                                work_prompt = load_work_prompt(work_prompt.path, language_name(target_lang)) \
                                    or work_prompt
                            if work_memory:
                                _save_memory(work_memory, runner.memory, work_dir, events, log)
                            if revise and finished and guard is not None:
                                backend = inpaint_backend or PipelineInpaintBackend(translator, config)
                                reuse, kept_back = await _inpaint_changes(finished, guard, backend, store, events, log,
                                                                          should_stop)
                                reviewed |= reuse
                                held.update(kept_back)
                        except BaseException:
                            for task in (review_task, proof_task):
                                if task is not None and not task.done():
                                    task.cancel()
                                    await asyncio.gather(task, return_exceptions=True)
                            raise
                    except AgentStopped:
                        raise asyncio.CancelledError("Agent workflow stopped by user")
                    if reviewer is not None:
                        _note_review(store, review)
                        for item in review:
                            if item.get("status") == "reviewed":
                                reviewed.add(_key(item["image"]))
                                log(f"[Agent] {item['page']}: inpaint reviewed, {item.get('filled', 0)} flat-filled, "
                                    f"{item.get('retried', 0)} redone, {item.get('flagged', 0)} flagged")
                            else:
                                log(f"[Agent] {item['page']}: inpaint {item.get('status')} "
                                    f"({item.get('reason') or item.get('error', '')})")
                        totals = reviewer.totals
                        log(f"[Agent] Inpaint review tokens: {totals['prompt_tokens']} prompt "
                            f"({totals['cached_tokens']} cached), {totals['completion_tokens']} completion, "
                            f"{totals['requests']} requests")
            except asyncio.CancelledError:  # the stop button: pages still working or in review are stopped
                events.emit("stopped", agent="workflow", stage="agent")
                raise
            finally:
                events.close()
            for outcome in chapter:
                log(f"[Agent] {outcome['page']}: {outcome.get('status')}, flags {outcome.get('flags', 0)}")
            outcomes.extend(chapter)
            if translated[0] != len(outcomes):  # pages that failed count as handled too
                translated[0] = len(outcomes)
                _report(progress, "translate", translated[0], len(paths))

        if not render:
            if guard is None:
                record_saved(paths)
            return []
        from ..utils.path_manager import find_json_path

        # Only pages the agent translated: a failed page, or one the run never reached, has only the
        # preprocess JSON, and the render would typeset its original text again.
        statuses = {_key(o["image"]): o.get("status") for o in outcomes if o.get("image")}
        errors = {_key(o["image"]): str(o.get("error") or "") for o in outcomes if o.get("image")}
        render_items, left_out = [], {}
        for image, cfg in images_with_configs:
            path = _image_path(image)
            status = statuses.get(_key(path)) or "not_reached"
            if status in RENDERED and json_guard.taken(find_json_path(path)):
                status = "edited"  # saved in the editor after the agent: the user's
            if status in RENDERED and _key(path) in held:  # its inpainted image is not up to its mask
                status = held[_key(path)]
            if status in RENDERED and find_json_path(path):
                render_items.append((image, cfg))
            else:
                left_out[path] = "no_json" if status in RENDERED else status
        if (revise or redo) and render_items:  # the output images are the user's exports: a copy first, or no render
            unsaved = _keep_outputs(translator, render_items, save_info, log)
            if unsaved:
                render_items = [item for item in render_items if _key(_image_path(item[0])) not in unsaved]
                for path in unsaved.values():
                    left_out[path] = "output_kept"
        log(f"[Agent] 4/4 Render: typesetting {len(render_items)} page(s)")
        if left_out:
            log("[Agent] Not rendered (no agent translation, or kept as it was): " + ", ".join(
                f"{os.path.basename(p)} ({s})" for p, s in left_out.items()))
        results: list = []
        render_events = _RenderEvents(chapter_logs)
        batch_progress.stage, batch_progress.total = "render", len(render_items)
        _report(progress, "render", 0, len(render_items))
        saved_hook = getattr(translator, "on_result_saved", None)
        translator.on_result_saved = render_events.saved
        guard_hook = getattr(translator, "json_save_guard", None)
        # a page saved in the editor while it is typeset keeps the user's JSON
        translator.json_save_guard = lambda image: not json_guard.taken(find_json_path(image))
        region_hook = getattr(translator, "load_text_region_hook", None)
        translator.load_text_region_hook = editor_box_center  # text where the editor puts it
        try:
            render_events.start([_image_path(image) for image, _ in render_items], left_out)
            # Reviewed pages reuse the inpainted image the review saved; the others
            # must not pick up an older one made before the agent changed the mask.
            for reuse in (True, False):
                batch = [item for item in render_items if (_key(_image_path(item[0])) in reviewed) == reuse]
                if batch:
                    _set_mode(translator, load_text=True, save_text=True, skip_txt_import=True,
                              ignore_existing_inpainted=not reuse, keep_empty_translations=True)
                    results.extend(await translator.translate_batch(batch, save_info=render_info))
                    batch_progress.offset += len(batch)
            render_events.finish(results)
        except asyncio.CancelledError:
            render_events.stopped()
            raise
        finally:
            translator.on_result_saved = saved_hook
            translator.json_save_guard = guard_hook
            translator.load_text_region_hook = region_hook
            render_events.close()
        if guard is None:  # what the run leaves, so the next run sees changes made in the editor
            record_saved(paths)
        return results + [_left_out_context(path, status, errors.get(_key(path), ""))
                          for path, status in left_out.items()]
    except asyncio.CancelledError:
        if not chapter_logs:  # stopped before a chapter started: the page would wait for it
            broadcast("workflow_stage", stage="stopped")
        raise
    except Exception as exc:
        if not chapter_logs:  # the API check or the preprocess failed
            broadcast("workflow_stage", stage="failed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        raise
    finally:
        users = json_guard.users_files()
        if users:
            log(f"[Agent] {len(users)} page(s) saved in the editor during the run were left as you saved them: "
                + ", ".join(sorted(os.path.basename(p) for p in users)))
        if guard is not None:  # also when stopped or failed: unfinished pages get their copies back
            try:
                # only the TXT files this run's preprocess exported (a check has none of its own)
                restored = guard.finish(getattr(translator, "_delete_original_txt_after_json_translation", None)
                                        if preprocess else None)
            except Exception:
                logger.exception("restoring the pages the run did not finish failed")
            else:
                if restored:
                    log(f"[Agent] {len(restored)} page(s) not finished: their translations from before the run "
                        f"are back: {', '.join(os.path.basename(p) for p in restored)}")
                if guard.not_restored:
                    log(f"[Agent] {len(guard.not_restored)} page(s) could not be put back as they were before the "
                        f"run (a file was in use): {', '.join(os.path.basename(p) for p in guard.not_restored)}. "
                        f"Their copies and the run journal stay; they are put back when the chapter is opened "
                        f"for the next run")
                    broadcast("workflow_stage", stage="restore_failed",
                              pages=[os.path.basename(p) for p in guard.not_restored])
        json_guard.reset()
        if isinstance(hooks, list) and batch_progress in hooks:
            hooks.remove(batch_progress)
        if font_task is not None and not font_task.done():  # stopped during the preprocess
            font_task.cancel()
        for name, value in saved.items():
            setattr(translator, name, value)
        if own_client and hasattr(client, "aclose"):
            try:
                await client.aclose()
            except Exception:
                logger.debug("agent: closing the API client failed", exc_info=True)
