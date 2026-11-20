"""The ``agent`` config section (design doc, "設定項")."""
from __future__ import annotations

from pydantic import BaseModel, Field


class AgentSettings(BaseModel):
    model: str = ""
    """Model name. Empty uses OPENAI_MODEL, like the OpenAI translators."""
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
    low_confidence_threshold: float = 0.7
    """Pages marked done below this confidence go to the review list."""
    request_timeout: int = 300
    """Seconds per API request."""
    retry_attempts: int = Field(default=3, ge=1)
    """Attempts per API candidate before moving to the next candidate."""
    memory_dir: str = ""
    """Glossary, characters and story summary. Empty keeps them in the chapter's
    ``manga_translator_work/agent/memory``; point several chapters at one folder to share them."""
    extra_instructions: str = ""
    """Extra instructions appended to the agent's system prompt."""
