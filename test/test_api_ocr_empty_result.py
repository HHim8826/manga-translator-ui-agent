"""Blank detector boxes must not trap API OCR in an endless retry loop (upstream #337)."""
import asyncio
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from manga_translator.ocr import model_api_ocr
from manga_translator.ocr.model_api_ocr import ModelGeminiOCR, ModelOpenAIOCR


def _make(cls):
    ocr = object.__new__(cls)
    ocr.logger = logging.getLogger("test-api-ocr")
    return ocr


def _run_once(monkeypatch):
    calls = []

    async def fake_run_with_api_candidates(*, operation, **kwargs):
        calls.append(kwargs)
        return await operation(SimpleNamespace(api_key="k", base_url="http://x", model_name="m"))

    monkeypatch.setattr(model_api_ocr, "run_with_api_candidates", fake_run_with_api_candidates)
    return calls


def _settings():
    return SimpleNamespace(api_key="k", candidates=[object()], strategy="sequential")


def _recognize(ocr, response):
    async def create(**kwargs):
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        models=SimpleNamespace(generate_content=create),
        close=lambda: None,
    )
    ocr._create_client = lambda api_key, base_url: client
    img = np.zeros((48, 64, 3), np.uint8)
    return asyncio.run(
        ocr._recognize_single(img, "prompt", runtime_settings=_settings(), custom_params_config=None)
    )


def _openai_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_openai_blank_region_returns_empty_without_retry(monkeypatch):
    calls = _run_once(monkeypatch)
    assert _recognize(_make(ModelOpenAIOCR), _openai_response("")) == ""
    assert len(calls) == 1


def test_openai_text_is_returned(monkeypatch):
    _run_once(monkeypatch)
    assert _recognize(_make(ModelOpenAIOCR), _openai_response("  セリフ \n")) == "セリフ"


def test_openai_missing_choices_is_still_an_error(monkeypatch):
    _run_once(monkeypatch)
    with pytest.raises(RuntimeError, match="choices"):
        _recognize(_make(ModelOpenAIOCR), SimpleNamespace(choices=[]))


def test_gemini_blank_region_returns_empty(monkeypatch):
    _run_once(monkeypatch)
    response = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[]))], text="")
    assert _recognize(_make(ModelGeminiOCR), response) == ""


def test_gemini_without_candidates_is_an_error(monkeypatch):
    _run_once(monkeypatch)
    with pytest.raises(RuntimeError, match="candidates"):
        _recognize(_make(ModelGeminiOCR), SimpleNamespace(candidates=[], text=""))
