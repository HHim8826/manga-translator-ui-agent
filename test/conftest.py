"""Session end for the tests (see ``_bootstrap`` for what every test module sets up first)."""
import _bootstrap  # noqa: F401, I001


def pytest_sessionfinish(session, exitstatus):
    _bootstrap.delete_widgets()
