"""The ``agent`` config section."""
from __future__ import annotations

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from pydantic_core import PydanticCustomError

MIN_CONTEXT_TOKENS = 32_000
"""The smallest ``max_context_tokens``: a page's first request (system prompt, tools, glossary,
memory and the page image) took about 16k tokens in real runs, and working a page adds about 11k."""


class AgentSettings(BaseModel):
    model: str = ""
    """Model name. Empty uses OPENAI_MODEL, like the OpenAI translators."""
    reasoning_effort: str = ""
    """``reasoning_effort`` of the translation agent's and the proofreader's requests (``low`` /
    ``medium`` / ``high`` / ``xhigh``, or any level the endpoint offers). Empty sends none, so the
    model's own default applies. An endpoint that rejects the field gets requests without it."""
    review_reasoning_effort: str = ""
    """The same for the recall check and the inpaint review."""
    max_context_tokens: int = Field(default=400_000, ge=MIN_CONTEXT_TOKENS)
    """Hard limit of one conversation."""
    compact_threshold_tokens: int = Field(default=300_000, ge=1, validate_default=True)
    """Compact the conversation at the next page boundary once a request is this large. Must be
    below ``max_context_tokens``, or the conversation would hit the limit before it is compacted."""
    keep_images_last_n_pages: int = Field(default=5, ge=1)
    """Older pages keep their text in the conversation but their images are dropped."""
    whole_chapter_conversation: bool = False
    """Carry the earlier pages' conversation (their images and tool results) to the next page, up to
    ``compact_threshold_tokens``. Off, each page starts from the memory and the last pages' translations
    (``AgentRunner.run_page``): every request resends the conversation, and on 131 pages of real runs
    the carried pages made the translation agent's requests 2.6 times as big (391k tokens a page; about
    150k starting each page afresh)."""
    page_image_max_side: int = Field(default=2048, ge=0)
    """Longest side of the whole-page images the translation agent and the recall check get (a zoom's
    crop is not limited); 0 sends them at full size. The message gives the page's own size, and boxes
    stay in page pixels. Vision APIs scale a larger image down on their side anyway (OpenAI's high
    detail fits it in 2048x2048), so a full-size scan only made every request bigger: each step resends
    the images of 5 to 9 pages (review 2026-11-24 P1)."""
    max_steps_per_page: int = Field(default=40, ge=1)
    """Model calls allowed for one page before it is flagged and the run moves on."""
    image_detail: str = "high"
    """``detail`` of page images: ``high`` (about 2.9k tokens per page) or ``low``."""
    double_read: str = "small_or_handwritten"
    """Which regions get a second, zoomed read: ``all`` / ``small_or_handwritten`` / ``none``."""
    recall_pass: bool = True
    """Before each page, a separate model call looks for text the detector
    missed; what it finds is snapped to the strokes and added as regions.
    The call for the next page runs while the agent works on the current one."""
    inpaint_review: bool = True
    """After the translation agent, a separate review conversation per page
    checks and fixes the inpainting; render then reuses its result."""
    review_while_translating: bool = True
    """Review each page's inpainting as soon as the translation agent finishes it, while the
    agent works on the next page; off waits for the whole chapter first."""
    fill_flat_areas: bool = True
    """In the review's first pass, areas on a flat background (plain bubbles and boxes) are
    filled with the paper color instead of using the model's result, unless the program
    checks find more problems in the fill; the model's result stays as a candidate."""
    inpaint_attempts_per_region: int = Field(default=3, ge=1)
    """New inpainting attempts the review may make per area (the first pass not counted)."""
    ops_per_region: int = Field(default=15, ge=1)
    """Review operations (views, mask edits, attempts) per area."""
    inpaint_review_steps: int = Field(default=30, ge=1)
    """Model calls allowed for one page's inpaint review."""
    proofread: bool = False
    """After the translation agent, a separate conversation proofreads the whole chapter's
    translations at once and fixes what is really wrong (off by default)."""
    proofread_steps: int = Field(default=20, ge=1)
    """Model calls allowed for the proofreading of one chapter."""
    choose_fonts: bool = True
    """The translation agent sees a sheet of the installed fonts that have every common character of
    the target language (Chinese, Japanese, Korean) and may give a region the one that looks like its
    original lettering (narration, handwriting, a sign); normal dialogue keeps the user's font."""
    fix_missing_glyphs: bool = True
    """After the agents, a region whose font has no glyph for one of its characters gets the most
    similar installed font that has them all. Every font change is checked by rendering the region
    with both fonts; a change that would move or resize the text is not made."""
    low_confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    """Pages marked done below this confidence go to the review list."""
    request_timeout: int = Field(default=300, ge=1)
    """Seconds per API request."""
    retry_attempts: int = Field(default=3, ge=1)
    """Attempts per API candidate before moving to the next candidate."""
    stream_requests: bool = True
    """Receive each reply as it is written (streaming). The reply starts within seconds, so a proxy
    that cuts a connection after a silent while (for example 120 s) does not cut a long reply. An API
    that refuses streaming, or streams without token counts, gets requests without it."""
    memory_dir: str = ""
    """One folder for the glossary, characters and story summary of every chapter. Empty keeps them
    in each chapter's ``manga_translator_work/agent/memory`` (shared through the work's memory, see
    ``share_memory``)."""
    share_memory: bool = True
    """The chapters of a work share what they learn: a chapter's first run starts with the glossary,
    characters and story summary of the chapters before it, and each chapter adds its own when it
    ends. The work's memory sits next to its prompt file (``<folder>/agent/<name>.memory/``); without
    a prompt file every chapter keeps its own (``work_memory.py``)."""
    show_page_on_start: bool = True
    """Switch to the AI Agent page when an AI Agent run starts."""
    use_work_prompt: bool = True
    """Use the translator's prompt file for the work (``translator.high_quality_prompt_path``, for
    example ``dict/<work>.yaml``): its instructions and glossary go into the agent's system prompt,
    and the program checks the glossary translations."""
    update_work_glossary: bool = True
    """After each chapter, add the new names and terms the agent recorded (and used in the final
    translations) to that prompt file, so later chapters and the other translators use them. Existing
    entries are never changed; YAML comments and layout stay; the user's own version is kept as
    ``.bak``, and the last versions in ``agent/backups/``."""
    revise_lock_user_edits: bool = True
    """When a chapter the AI Agent finished is checked again, the regions the user changed in the
    editor after that run (their text or place), or drew there, stay as the user left them: the agent,
    the proofreader and the font check leave them. Off lets the agent change them too."""
    extra_instructions: str = ""
    """Extra instructions appended to the agent's system prompt."""

    @field_validator("compact_threshold_tokens")
    @classmethod
    def _compact_below_limit(cls, value: int, info: ValidationInfo) -> int:
        limit = info.data.get("max_context_tokens")  # validated before, unless it was invalid
        if limit is not None and value >= limit:
            raise PydanticCustomError("compact_not_below_limit",
                                      "must be below max_context_tokens ({limit})", {"limit": limit})
        return value
