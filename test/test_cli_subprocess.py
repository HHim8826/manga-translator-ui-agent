"""The CLI's subprocess mode (the project review of 2026-11-22): P1-04, two chapters' ``001.png`` were
both reported done and only one output was left; P2-05, ``--memory-limit`` / ``--memory-percent``
were only in the help text."""
import asyncio
import os
import queue
import threading
from types import SimpleNamespace

import pytest

import manga_translator
from manga_translator.mode import subprocess_manager as sm


class FakeTranslator:
    batches = []
    save_infos = []

    def __init__(self, params):
        self.params = params

    async def translate_batch(self, items, save_info=None):
        FakeTranslator.batches.append([os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
                                       for p, _ in items])
        FakeTranslator.save_infos.append(save_info)
        return [SimpleNamespace(image_name=p, success=True, skipped=False, translation_error=None)
                for p, _ in items]


@pytest.fixture
def fake_translator(monkeypatch):
    FakeTranslator.batches, FakeTranslator.save_infos = [], []
    monkeypatch.setattr(manga_translator, "MangaTranslator", FakeTranslator)
    return FakeTranslator


def _pages(tmp_path):
    root = tmp_path / "manga"
    paths = []
    for chapter in ("ch01", "ch02"):
        (root / chapter).mkdir(parents=True)
        path = root / chapter / "001.png"
        path.write_bytes(b"png")
        paths.append(str(path))
    return str(root), paths


def test_pages_of_two_chapters_with_the_same_name_keep_their_chapters(tmp_path, fake_translator):
    root, paths = _pages(tmp_path)
    assert sm.same_output_paths(paths, [root], None) == {}  # <output>/manga/ch01/001.png, .../ch02/001.png
    results = queue.Queue()
    sm.worker_translate_batch(paths, str(tmp_path / "out"), False, True, {"cli": {}}, 0, 0, results, [root])
    assert results.get_nowait()["completed"] == paths
    assert fake_translator.save_infos[0]["input_folders"] == {os.path.normpath(root)}


def test_pages_that_would_be_written_to_one_path_are_not_started(tmp_path):
    _root, paths = _pages(tmp_path)
    assert sm.same_output_paths(paths, [], None) == {paths[1]: paths[0]}  # given as files
    jpg = os.path.join(os.path.dirname(paths[0]), "001.jpg")
    assert sm.same_output_paths([paths[0], jpg], [], "png") == {jpg: paths[0]}  # both saved as 001.png
    assert sm.same_output_paths([paths[0], jpg], [], None) == {}
    archives = [os.path.join(str(tmp_path), name, "original_images", "0000_a.png") for name in ("第1話", "第2話")]
    assert sm.same_output_paths(archives, [], None) == {}  # each archive has its own folder


def _inline(monkeypatch):
    """The workers run in a thread of this process (a test cannot pickle the fake translator)."""
    started = []

    class Process:
        def __init__(self, target, args):
            self.thread = threading.Thread(target=target, args=args)
            self.files = args[0]

        def start(self):
            started.append(len(self.files))
            self.thread.start()

        def join(self, timeout=None):
            self.thread.join(timeout)

        def is_alive(self):
            return False

    monkeypatch.setattr(sm.multiprocessing, "Process", Process)
    monkeypatch.setattr(sm.multiprocessing, "Queue", queue.Queue)
    return started


def test_the_memory_check_reads_the_process_and_the_system(monkeypatch):
    monkeypatch.setattr(sm, "get_memory_usage_mb", lambda: 2048)
    monkeypatch.setattr(sm, "get_system_memory_percent", lambda: 50)
    assert sm.over_memory_limit(1024, 0) == "2048 MB > 1024 MB"
    assert sm.over_memory_limit(4096, 0) is None and sm.over_memory_limit(0, 60) is None
    assert sm.over_memory_limit(0, 40) == "50% > 40%"


def test_the_subprocess_restarts_once_when_it_goes_over_its_memory_limit(tmp_path, fake_translator, monkeypatch):
    root = tmp_path / "manga" / "ch01"
    root.mkdir(parents=True)
    paths = []
    for i in range(1, 4):
        (root / f"{i:03d}.png").write_bytes(b"png")
        paths.append(str(root / f"{i:03d}.png"))
    started = _inline(monkeypatch)
    readings = iter(["2048 MB > 1024 MB", None, None])  # over the limit after the first page, then below it
    monkeypatch.setattr(sm, "over_memory_limit", lambda mb, percent: next(readings))
    counts = asyncio.run(sm.translate_with_subprocess(paths, str(tmp_path / "out"), {"cli": {"batch_size": 1}},
                                                      False, True, memory_limit_mb=1024, memory_limit_percent=0,
                                                      batch_per_restart=0, input_folders=[str(root)]))
    assert counts == (3, 0, 0)
    assert started == [3, 2]  # the second worker goes on with the two pages left
    assert fake_translator.batches == [["ch01/001.png"], ["ch01/002.png"], ["ch01/003.png"]]


def test_without_a_memory_limit_the_batch_is_translated_at_once(tmp_path, fake_translator, monkeypatch):
    root, paths = _pages(tmp_path)
    started = _inline(monkeypatch)
    monkeypatch.setattr(sm, "over_memory_limit", lambda mb, percent: pytest.fail("no limit: no reading"))
    counts = asyncio.run(sm.translate_with_subprocess(paths + [paths[0].replace("ch01", "ch03")],
                                                      str(tmp_path / "out"), {"cli": {}}, False, True,
                                                      memory_limit_mb=0, memory_limit_percent=0,
                                                      batch_per_restart=0, input_folders=[]))
    # given as files: ch02's and ch03's 001.png would go where ch01's does
    assert counts == (1, 0, 2) and started == [1]
    assert fake_translator.batches == [["ch01/001.png"]]
