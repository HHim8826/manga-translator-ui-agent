"""The AI agent workflow mode: preprocess -> agent -> render, on one MangaTranslator.

1. Preprocess with the existing "Export Original Text" mode: detection, local
   OCR and the refined mask go into ``json/*_translations.json``.
2. The agent works through each chapter folder in page order and writes its
   results back into those JSON files (plus sidecars and an event log).
3. Render with the existing "Import Translation and Render" mode, without the
   TXT import (it would overwrite the agent's JSON) and without reusing older
   inpainted images (the agent may have changed the mask).

The editor opens the same JSON files afterwards.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .events import EventLog, new_run_id
from .runner import AgentRunner, AgentStopped
from .settings import AgentSettings
from .store import AgentStore

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
) -> list:
    settings = agent_settings if isinstance(agent_settings, AgentSettings) else AgentSettings(**(agent_settings or {}))
    if not images_with_configs:
        return []
    config = images_with_configs[0][1]
    paths = [_image_path(image) for image, _ in images_with_configs]
    saved = {name: getattr(translator, name, False) for name in MODE_FLAGS}
    try:
        if preprocess:
            log("[Agent] 1/3 Preprocess: detection, local OCR and masks")
            _set_mode(translator, template=True, save_text=True)
            await translator.translate_batch(images_with_configs, save_info=save_info)
        if should_stop():
            raise asyncio.CancelledError("Agent workflow stopped by user")

        if client is None:
            from .client import build_client
            client = build_client(config, settings)
        target_lang = getattr(getattr(config, "translator", None), "target_lang", "") or "CHT"
        log(f"[Agent] 2/3 Agent: {len(paths)} page(s), model {getattr(client, 'model_label', '')}")
        outcomes: List[dict] = []
        for work_dir, pages in group_by_chapter(paths):
            store = AgentStore(work_dir, settings.memory_dir)
            run_id = new_run_id()
            events = EventLog(store.run_log_path(run_id), run_id)
            log(f"[Agent] {os.path.basename(os.path.dirname(work_dir))}: event log {events.path}")
            try:
                runner = AgentRunner(settings, client, store, events, pages, target_lang, should_stop)
                try:
                    chapter = await runner.run()
                except AgentStopped:
                    raise asyncio.CancelledError("Agent workflow stopped by user")
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
        log(f"[Agent] 3/3 Render: inpainting and typesetting {len(render_items)} page(s)")
        _set_mode(translator, load_text=True, save_text=True, skip_txt_import=True, ignore_existing_inpainted=True)
        return await translator.translate_batch(render_items, save_info=save_info)
    finally:
        for name, value in saved.items():
            setattr(translator, name, value)
