"""The ``agent`` config section (design doc, "設定項")."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentSettings(BaseModel):
    model: str = ""
    """Model name. Empty uses OPENAI_MODEL, like the OpenAI translators."""
    reasoning_effort: str = ""
    """``reasoning_effort`` of the translation agent's and the proofreader's requests (``low`` /
    ``medium`` / ``high`` / ``xhigh``, or any level the endpoint offers). Empty sends none, so the
    model's own default applies. An endpoint that rejects the field gets requests without it."""
    review_reasoning_effort: str = ""
    """The same for the recall check and the inpaint review."""
    max_context_tokens: int = 400_000
    """Hard limit of one conversation."""
    compact_threshold_tokens: int = 300_000
    """Compact the conversation at the next page boundary once a request is this large."""
    keep_images_last_n_pages: int = Field(default=5, ge=1)
    """Older pages keep their text in the conversation but their images are dropped."""
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
    low_confidence_threshold: float = 0.7
    """Pages marked done below this confidence go to the review list."""
    request_timeout: int = 300
    """Seconds per API request."""
    retry_attempts: int = Field(default=3, ge=1)
    """Attempts per API candidate before moving to the next candidate."""
    memory_dir: str = ""
    """Glossary, characters and story summary. Empty keeps them in the chapter's
    ``manga_translator_work/agent/memory``; point several chapters at one folder to share them."""
    use_work_prompt: bool = True
    """Use the translator's prompt file for the work (``translator.high_quality_prompt_path``, for
    example ``dict/<work>.yaml``): its instructions and glossary go into the agent's system prompt,
    and the program checks the glossary translations. The file is only read."""
    extra_instructions: str = ""
    """Extra instructions appended to the agent's system prompt."""
