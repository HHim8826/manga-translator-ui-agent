"""The files a page's editor document is read from, as stamps (path, mtime_ns, size).

A document read earlier (a prefetched neighbour page, or the page open in the editor) is only as
new as these files were then: when an AI Agent run, the batch management or another program writes
one of them later, the copy in memory is older than the disk. The editor compares the stamps to
notice that, so it does not show the old copy or save it over the newer files."""
from __future__ import annotations

import os
from typing import Optional

from manga_translator.utils.path_manager import (
    find_inpainted_path,
    find_json_path,
    find_paint_overlay_path,
    find_work_image_path,
)

Stamp = Optional[tuple]


def file_stamp(path: Optional[str]) -> Stamp:
    """(path, mtime_ns, size) of the file; (path, None, None) when it is not there; None without a
    path."""
    if not path:
        return None
    key = os.path.normcase(os.path.abspath(path))
    try:
        stat = os.stat(path)
    except OSError:
        return (key, None, None)
    return (key, stat.st_mtime_ns, stat.st_size)


def _found(find, source_path: str) -> Optional[str]:
    try:
        return find(source_path)
    except Exception:
        return None


def json_stamp(source_path: str) -> Stamp:
    """The stamp of the page's project JSON (None while the page has none)."""
    return file_stamp(_found(find_json_path, source_path))


def document_files(source_path: str) -> tuple:
    """The stamps of every file the editor reads for the page."""
    if not source_path:
        return ()
    return (
        ("source", file_stamp(source_path)),
        ("work_image", file_stamp(_found(find_work_image_path, source_path))),
        ("json", json_stamp(source_path)),
        ("inpainted", file_stamp(_found(find_inpainted_path, source_path))),
        ("paint_overlay", file_stamp(_found(find_paint_overlay_path, source_path))),
    )
