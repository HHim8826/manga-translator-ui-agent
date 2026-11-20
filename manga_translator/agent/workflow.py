"""The AI agent workflow mode: preprocess -> agent -> render, on one MangaTranslator.

1. Preprocess with the existing "Export Original Text" mode: detection, local
   OCR and the refined mask go into ``json/*_translations.json``.
2. The agent works through each chapter folder in page order and writes its
   results back into those JSON files (plus sidecars and an event log).
3. Inpaint review (``inpaint_review``): per finished page, the program inpaints
   block by block and runs the inpainting checks, and a separate review
   conversation fixes what is wrong. The chosen result is saved as the page's
   inpainted image and the JSON mask is updated.
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
from .inpaint_review import InpaintReviewer
from .runner import AgentRunner, AgentStopped, language_name
from .settings import AgentSettings
from .store import AgentStore
from .work_prompt import load_work_prompt

logger = logging.getLogger("manga_translator.agent")

MODE_FLAGS = ("template", "save_text", "load_text", "translate_json_only", "generate_and_export",
              "colorize_only", "upscale_only", "inpaint_only", "replace_translation", "export_from_local_json",
              "skip_txt_import", "ignore_existing_inpainted")


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


async def _review_queue(reviewer: InpaintReviewer, queue: "asyncio.Queue") -> List[dict]:
    """Review pages as they arrive until ``None``."""
    outcomes: List[dict] = []
    while True:
        path = await queue.get()
        if path is None:
            return outcomes
        outcomes.extend(await reviewer.run([path]))


def group_by_chapter(paths: Sequence[str]) -> List[Tuple[str, List[str]]]:
    """(work dir, pages) per folder, folders in first-seen order, pages in input order."""
    from ..utils.path_manager import get_work_dir

    groups: Dict[str, List[str]] = {}
    for path in paths:
        groups.setdefault(get_work_dir(path), []).append(path)
    return list(groups.items())


async def run_agent_workflow(
    translator,
    images_with_configs: List[tuple],
    save_info: Optional[dict] = None,
    *,
    agent_settings: Union[AgentSettings, dict, None] = None,
    should_stop: Callable[[], bool] = lambda: False,
    log: Callable[[str], Any] = logger.info,
    client=None,
    preprocess: bool = True,
    render: bool = True,
    inpaint_backend=None,
) -> list:
    settings = agent_settings if isinstance(agent_settings, AgentSettings) else AgentSettings(**(agent_settings or {}))
    if not images_with_configs:
        return []
    config = images_with_configs[0][1]
    paths = [_image_path(image) for image, _ in images_with_configs]
    saved = {name: getattr(translator, name, False) for name in MODE_FLAGS}
    try:
        if preprocess:
            log("[Agent] 1/4 Preprocess: detection, local OCR and masks")
            _set_mode(translator, template=True, save_text=True)
            await translator.translate_batch(images_with_configs, save_info=save_info)
        if should_stop():
            raise asyncio.CancelledError("Agent workflow stopped by user")

        if client is None:
            from .client import build_client
            client = build_client(config, settings)
        target_lang = getattr(getattr(config, "translator", None), "target_lang", "") or "CHT"
        log(f"[Agent] 2/4 Agent: {len(paths)} page(s), model {getattr(client, 'model_label', '')}")
        work_prompt = None
        if settings.use_work_prompt:
            prompt_path = getattr(getattr(config, "translator", None), "high_quality_prompt_path", None) or ""
            work_prompt = load_work_prompt(prompt_path, language_name(target_lang))
            if work_prompt is not None:
                log(f"[Agent] Work prompt: {work_prompt.name}, {len(work_prompt.terms)} glossary term(s)")
        outcomes: List[dict] = []
        reviewed: set = set()
        for work_dir, pages in group_by_chapter(paths):
            store = AgentStore(work_dir, settings.memory_dir)
            run_id = new_run_id()
            events = EventLog(store.run_log_path(run_id), run_id)
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
                    if review_task is not None and outcome.get("status") in ("done", "incomplete"):
                        queue.put_nowait(outcome["image"])

                runner = AgentRunner(settings, client, store, events, pages, target_lang, should_stop,
                                     on_page_done=page_done, work_prompt=work_prompt)
                review: List[dict] = []
                try:
                    try:
                        chapter = await runner.run()
                    except BaseException:
                        if review_task is not None:
                            review_task.cancel()
                            await asyncio.gather(review_task, return_exceptions=True)
                        raise
                    if review_task is not None:
                        queue.put_nowait(None)
                        review = await review_task
                    elif reviewer is not None:
                        to_review = [o["image"] for o in chapter if o.get("status") in ("done", "incomplete")]
                        if to_review:
                            log(f"[Agent] 3/4 Inpaint review: {len(to_review)} page(s)")
                            review = await reviewer.run(to_review)
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
            finally:
                events.close()
            for outcome in chapter:
                log(f"[Agent] {outcome['page']}: {outcome.get('status')}, flags {outcome.get('flags', 0)}")
                if outcome.get("status") in ("done", "incomplete"):
                    # Same as "Translate JSON Only": the exported original TXT would
                    # otherwise be imported over the agent's JSON later.
                    translator._delete_original_txt_after_json_translation(outcome["image"])
            outcomes.extend(chapter)

        if not render:
            return []
        from ..utils.path_manager import find_json_path

        render_items = [(image, cfg) for image, cfg in images_with_configs if find_json_path(_image_path(image))]
        log(f"[Agent] 4/4 Render: typesetting {len(render_items)} page(s)")
        results: list = []
        # Reviewed pages reuse the inpainted image the review saved; the others
        # must not pick up an older one made before the agent changed the mask.
        for reuse in (True, False):
            batch = [item for item in render_items if (_key(_image_path(item[0])) in reviewed) == reuse]
            if batch:
                _set_mode(translator, load_text=True, save_text=True, skip_txt_import=True,
                          ignore_existing_inpainted=not reuse)
                results.extend(await translator.translate_batch(batch, save_info=save_info))
        return results
    finally:
        for name, value in saved.items():
            setattr(translator, name, value)
