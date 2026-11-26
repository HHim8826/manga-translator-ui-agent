import _bootstrap  # noqa: F401

import subprocess
import sys


ROOT = _bootstrap.ROOT


def test_main_import_suppresses_qfluentwidgets_promotion():
    result = subprocess.run(
        [sys.executable, "-c", "import desktop_qt_ui.main"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "QFluentWidgets Pro" not in result.stdout


# the end of main(): the QApplication is its first local, the window comes after it
_EXIT = """
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import desktop_qt_ui.main as entry
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget


def main():
    app = QApplication(sys.argv)
    from ui.main_page.pages.agents_page import AgentsPage
    main_window = QWidget()
    QVBoxLayout(main_window).addWidget(AgentsPage(lambda key, **kw: key, main_window))
    main_window.resize(900, 700)
    main_window.show()
    app.processEvents()
    main_window.close()
    entry.delete_windows()
    return 0


raise SystemExit(main())
"""


def test_the_program_ends_without_a_crash_after_its_window_was_shown():
    """Review 2026-11-26: the tests' Python exited with 0xC0000409 after every test passed. A stand-in
    for main() (the AI Agent page in a window) ended so 5 times of 6 while the window outlived the
    QApplication; ``delete_windows`` deletes it first."""
    result = subprocess.run([sys.executable, "-c", _EXIT], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
