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

The editor opens the same JSON files afterwards.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .events import EventLog, new_run_id
from .font_styles import load_work_styles, save_work_styles, work_fonts_path
from .inpaint_review import InpaintReviewer
from .proofread import ProofPage, Proofreader
from .resume import RunGuard, record_saved
from .runner import AgentRunner, AgentStopped, language_name
from .settings import AgentSettings
from .store import AgentStore
from .work_glossary import update_work_glossary
from .work_memory import bootstrap, save_chapter, seed_chapter, work_memory_dir
from .work_prompt import load_work_prompt, resolve_path

logger = logging.getLogger("manga_translator.agent")

MODE_FLAGS = ("template", "save_text", "load_text", "translate_json_only", "generate_and_export",
              "colorize_only", "upscale_only", "inpaint_only", "replace_translation", "export_from_local_json",
              "skip_txt_import", "ignore_existing_inpainted", "keep_empty_translations")
RENDERED = ("done", "incomplete")
"""Page statuses that are typeset; the other pages have no agent translation."""


def _image_path(image: Any) -> str:
    if isinstance(image, (str, os.PathLike)):
        return os.fspath(image)
    return getattr(image, "name", "") or ""


def _set_mode(translator, **flags: bool) -> None:
    for name in MODE_FLAGS:
        setattr(translator, name, bool(flags.get(name, False)))


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


class PipelineInpaintBackend:
    """The translator's inpainting models and text detector, for the inpaint review."""

    EXTRA_METHODS = ("lama_large", "flux2-klein")

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

    def save(self, image_path: str, image) -> None:
        self.translator._save_inpainted_image(image_path, image)


async def _proofread(proofreader: Proofreader, pages: List[str], events: EventLog, log) -> dict:
    """The proofreading pass; an API failure is logged and the chapter goes on to render as it is."""
    try:
        result = await proofreader.run(pages)
    except (AgentStopped, asyncio.CancelledError):
        raise
    except Exception as exc:
        logger.exception("proofreading failed")
        events.emit("error", agent="proofread", error=f"proofreading failed: {type(exc).__name__}: {str(exc)[:300]}")
        return {"status": "error", "error": str(exc)}
    totals = proofreader.totals
    log(f"[Agent] Proofread: {result.get('revised', 0)} revised, {result.get('flagged', 0)} flagged, "
        f"{totals['requests']} requests, {totals['prompt_tokens']} prompt tokens ({totals['cached_tokens']} cached)")
    return result


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
        seeded = seed_chapter(store.memory(), folder)
    except Exception as exc:
        logger.exception("reading the work's memory failed")
        log(f"[Agent] {chapter}: the work's memory was not read ({type(exc).__name__}: {exc})")
        return None
    if not seeded:
        return None
    log(f"[Agent] {chapter}: starts with the work's memory: {seeded['glossary']} term(s), "
        f"{seeded['characters']} character(s), story summary of {seeded['story']} characters")
    return {**seeded, "from": earlier}


def _save_memory(folder: str, memory, work_dir: str, events: EventLog, log) -> None:
    """Merge the chapter's memory into the work's; a failure is logged and the chapter goes on."""
    chapter = os.path.basename(os.path.dirname(os.path.abspath(work_dir)))
    try:
        changes = save_chapter(memory, folder, chapter)
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
            events.emit("error", page=os.path.splitext(os.path.basename(path))[0], agent="fonts",
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
        for work_dir, pages in group_by_chapter(list(paths) + list(left_out)):
            log = self._log(pages[0])
            if log is not None:
                skipped = {os.path.splitext(os.path.basename(p))[0]: left_out[p]
                           for p in pages if _key(p) not in rendered and p in left_out}
                log.emit("render_start", agent="render", pages=sum(1 for p in pages if _key(p) in rendered),
                         **({"skipped": skipped} if skipped else {}))

    def saved(self, ctx) -> None:  # the translator's on_result_saved hook
        from ..utils.path_manager import find_inpainted_path

        image = _image_path(getattr(ctx, "image_name", "") or "")
        if not image:
            return
        log = self._log(image)
        if log is not None:
            log.emit("render_done", page=os.path.splitext(os.path.basename(image))[0], agent="render",
                     image=image, output=getattr(ctx, "output_path", None), inpainted=find_inpainted_path(image))

    def stopped(self) -> None:
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


def _left_out_context(path: str, status: str, error: str = ""):
    """A result for a page the render left out, so the batch summary names it: a failure with its
    reason when the agent failed on it, else a skipped page (stopped, not reached)."""
    from ..utils import Context

    ctx = Context()
    ctx.image_name = path
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
) -> list:
    settings = agent_settings if isinstance(agent_settings, AgentSettings) else AgentSettings(**(agent_settings or {}))
    if not images_with_configs:
        return []
    config = images_with_configs[0][1]
    paths = [_image_path(image) for image, _ in images_with_configs]
    saved = {name: getattr(translator, name, False) for name in MODE_FLAGS}
    target_lang = getattr(getattr(config, "translator", None), "target_lang", "") or "CHT"
    font_task: Optional[asyncio.Future] = None
    guard: Optional[RunGuard] = None
    """Copies of the pages from before the run: the pages it does not finish get them back."""
    preprocess_failed: Dict[str, str] = {}
    # progress(stage, done, total): ``preprocess`` and ``render`` count the translator's pages,
    # ``translate`` the pages the agent finished, ``proofread`` starts a chapter's proofreading
    batch_progress = _BatchProgress(progress, len(paths))
    hooks = getattr(translator, "_progress_hooks", None)
    if progress is not None and isinstance(hooks, list):
        hooks.append(batch_progress)
    try:
        if fonts is None:  # scanning the installed fonts takes seconds: do it while the pages are preprocessed
            font_task = asyncio.ensure_future(_font_service(translator, config, settings, target_lang, log))
        if client is None:  # before the preprocess: a wrong API setting should not wait for it
            from .client import build_client, check_api
            client = build_client(config, settings)
            log(f"[Agent] API check: model {getattr(client, 'model_label', '')}")
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
            preprocessed = await translator.translate_batch(images_with_configs, save_info=save_info)
            batch_progress.stage = None
            preprocess_failed = guard.preprocess_failures(preprocessed)
            if preprocess_failed:
                log("[Agent] Preprocess failed, the agent leaves these pages as they were: " + "; ".join(
                    f"{os.path.basename(p)} ({why})" for p, why in preprocess_failed.items()))
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
        breaks = line_break_rules(config)
        if breaks:
            log("[Agent] AI line breaks: on (translations get [BR] line breaks by the source line count)")
        outcomes: List[dict] = []
        translated = [0]
        _report(progress, "translate", 0, len(paths))
        reviewed: set = set()
        chapter_logs: Dict[str, Tuple[str, str]] = {}
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
            log(f"[Agent] {os.path.basename(os.path.dirname(work_dir))}: event log {events.path}")
            if hasattr(client, "session_id"):  # one session per chapter run on the API side
                client.session_id = f"manga-translator-{run_id}-{uuid.uuid4().hex[:8]}"
            try:
                reviewer = None
                if settings.inpaint_review:
                    backend = inpaint_backend or PipelineInpaintBackend(translator, config)
                    reviewer = InpaintReviewer(settings, client, store, events, target_lang, backend, should_stop,
                                               save_image=getattr(backend, "save", None))
                queue: asyncio.Queue = asyncio.Queue()
                review_task: Optional[asyncio.Future] = None
                if reviewer is not None and settings.review_while_translating:
                    log("[Agent] 3/4 Inpaint review: each page as soon as the agent finishes it")
                    review_task = asyncio.ensure_future(_review_queue(reviewer, queue))

                def page_done(outcome: dict) -> None:
                    if guard is not None:
                        guard.page_finished(outcome["image"])
                    translated[0] += 1
                    _report(progress, "translate", translated[0], len(paths))
                    if review_task is not None and outcome.get("status") in ("done", "incomplete"):
                        queue.put_nowait(outcome["image"])

                use_styles = work_prompt is not None and settings.choose_fonts and font_service is not None
                styles_path = work_fonts_path(work_prompt.path) if use_styles else ""
                runner = AgentRunner(settings, client, store, events, pages, target_lang, should_stop,
                                     on_page_done=page_done, work_prompt=work_prompt,
                                     fonts=font_service if settings.choose_fonts else None, line_breaks=breaks,
                                     font_styles=_work_styles(styles_path, font_service, log) if styles_path else None,
                                     run_info={"chapter": os.path.basename(os.path.dirname(work_dir)), "batch": batch_id,
                                               **({"work_memory": seeded} if seeded else {})},
                                     preprocess_failed=preprocess_failed)
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
                                                      work_prompt, should_stop, line_breaks=bool(breaks))
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
                    except BaseException:
                        for task in (review_task, proof_task):
                            if task is not None and not task.done():
                                task.cancel()
                                await asyncio.gather(task, return_exceptions=True)
                        raise
                except AgentStopped:
                    raise asyncio.CancelledError("Agent workflow stopped by user")
                if reviewer is not None:
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
                if outcome.get("status") in ("done", "incomplete"):
                    # Same as "Translate JSON Only": the exported original TXT would
                    # otherwise be imported over the agent's JSON later.
                    translator._delete_original_txt_after_json_translation(outcome["image"])
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
            if status in RENDERED and find_json_path(path):
                render_items.append((image, cfg))
            else:
                left_out[path] = "no_json" if status in RENDERED else status
        log(f"[Agent] 4/4 Render: typesetting {len(render_items)} page(s)")
        if left_out:
            log("[Agent] Not rendered (no agent translation): " + ", ".join(
                f"{os.path.basename(p)} ({s})" for p, s in left_out.items()))
        results: list = []
        render_events = _RenderEvents(chapter_logs)
        batch_progress.stage, batch_progress.total = "render", len(render_items)
        _report(progress, "render", 0, len(render_items))
        saved_hook = getattr(translator, "on_result_saved", None)
        translator.on_result_saved = render_events.saved
        try:
            render_events.start([_image_path(image) for image, _ in render_items], left_out)
            # Reviewed pages reuse the inpainted image the review saved; the others
            # must not pick up an older one made before the agent changed the mask.
            for reuse in (True, False):
                batch = [item for item in render_items if (_key(_image_path(item[0])) in reviewed) == reuse]
                if batch:
                    _set_mode(translator, load_text=True, save_text=True, skip_txt_import=True,
                              ignore_existing_inpainted=not reuse, keep_empty_translations=True)
                    results.extend(await translator.translate_batch(batch, save_info=save_info))
                    batch_progress.offset += len(batch)
        except asyncio.CancelledError:
            render_events.stopped()
            raise
        finally:
            translator.on_result_saved = saved_hook
            render_events.close()
        if guard is None:  # what the run leaves, so the next run sees changes made in the editor
            record_saved(paths)
        return results + [_left_out_context(path, status, errors.get(_key(path), ""))
                          for path, status in left_out.items()]
    finally:
        if guard is not None:  # also when stopped or failed: unfinished pages get their copies back
            try:
                restored = guard.finish(getattr(translator, "_delete_original_txt_after_json_translation", None))
            except Exception:
                logger.exception("restoring the pages the run did not finish failed")
            else:
                if restored:
                    log(f"[Agent] {len(restored)} page(s) not finished: their translations from before the run "
                        f"are back: {', '.join(os.path.basename(p) for p in restored)}")
        if isinstance(hooks, list) and batch_progress in hooks:
            hooks.remove(batch_progress)
        if font_task is not None and not font_task.done():  # stopped during the preprocess
            font_task.cancel()
        for name, value in saved.items():
            setattr(translator, name, value)
