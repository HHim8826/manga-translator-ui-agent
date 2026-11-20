"""A runaway text box must be skipped instead of allocating gigabytes and failing the page."""
import numpy as np

from manga_translator.rendering.text_render import _render


def test_guard_threshold():
    assert not _render._text_canvas_too_large(2000, 2000, 40, "ok")
    assert _render._text_canvas_too_large(49971, 32768, 8000, "runaway")


def test_horizontal_render_skips_oversized_canvas(monkeypatch):
    allocations = []
    real_zeros = np.zeros

    def tracking_zeros(shape, *args, **kwargs):
        allocations.append(shape)
        return real_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(_render, "_MAX_TEXT_CANVAS_PIXELS", 10)
    monkeypatch.setattr(_render.np, "zeros", tracking_zeros)
    result = _render.put_text_horizontal(
        24, "テスト", 200, 60, "center", False, (0, 0, 0), (255, 255, 255), "CHT", True, 1.0, None, 1
    )
    assert result is None
    assert not any(len(s) == 3 and s[2] == 4 for s in allocations if isinstance(s, tuple))


def test_vertical_render_skips_oversized_canvas(monkeypatch):
    monkeypatch.setattr(_render, "_MAX_TEXT_CANVAS_PIXELS", 10)
    result = _render.put_text_vertical(
        24, "テスト", 200, "center", (0, 0, 0), (255, 255, 255), 1.0, None, 1
    )
    assert result is None


def test_normal_render_still_works():
    result = _render.put_text_horizontal(
        24, "テスト", 200, 60, "center", False, (0, 0, 0), (255, 255, 255), "CHT", True, 1.0, None, 1
    )
    assert result is not None and result.shape[2] == 4
