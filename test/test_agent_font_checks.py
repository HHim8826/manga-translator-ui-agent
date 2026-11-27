"""Review 2026-11-28 B16: the font catalog did not find a face by the name Qt resolved it to, the
font and size checks rendered with an AI renderer (a paid image request per region), light text was
drawn on a white page (nothing to measure), and the extent across a region's lines took the band
that starts last for the one that ends last."""
import asyncio
from types import SimpleNamespace

import numpy as np

from manga_translator.agent import fonts
from manga_translator.agent.render_check import lines_across


def _face(family, style="Regular", weight=400):
    return fonts.FontFace("x.ttf", 0, family, style, weight, (family,), frozenset())


def test_a_face_is_found_by_the_name_qt_resolved_it_to():
    regular, bold = _face("Shin Go"), _face("Shin Go", "Bold", 700)
    resolved = {"Shin Go": "Shin Go [Foundry]", "Shin Go::Bold": "Shin Go [Foundry]::Bold"}
    catalog = fonts.FontCatalog([regular, bold], resolve=resolved.get)
    assert regular.value == "Shin Go [Foundry]"
    assert catalog.find("Shin Go [Foundry]") is regular and catalog.find("Shin Go [Foundry]::Bold") is bold
    assert catalog.find("Shin Go") is regular  # its own name as before


def test_the_checks_render_with_the_local_renderer():
    from manga_translator.config import Config, Renderer

    config = Config()
    config.render.renderer = Renderer.openai_renderer
    backend = fonts.PipelineFontBackend(None, config)
    assert backend.config.render.renderer == Renderer.default
    assert config.render.renderer == Renderer.openai_renderer  # the user's config is left
    local = Config()
    assert fonts.PipelineFontBackend(None, local).config is local


def test_light_text_is_measured_on_a_dark_page(monkeypatch):
    from manga_translator import rendering

    async def dispatch(canvas, blocks, config, img, **kwargs):  # draws the block in its color
        out = canvas.copy()
        out[10:20, 10:30] = blocks[0].fg
        return out

    monkeypatch.setattr(rendering, "dispatch", dispatch)
    backend = fonts.PipelineFontBackend(None, SimpleNamespace(render=SimpleNamespace(renderer="default")))
    page = np.full((40, 40, 3), 128, np.uint8)
    backend._images = {"p.png": (page, None)}
    backend._drafts = {"p.png": (None, True, False)}
    for fg in ((255, 255, 255), (0, 0, 0)):
        block = SimpleNamespace(fg=fg, get_font_colors=lambda fg=fg: (fg, (0, 0, 0)))
        out = asyncio.run(backend.render_alone("p.png", block))
        assert (out[10:20, 10:30] == 0).all() and (out[0:5, 0:5] == 255).all(), fg  # dark text on white


def test_the_extent_across_lines_ends_where_the_last_one_ends():
    quads = [[[10, 0], [100, 0], [100, 5], [10, 5]], [[20, 10], [50, 10], [50, 15], [20, 15]]]
    assert lines_across(quads, "v") == (1, 90.0)
