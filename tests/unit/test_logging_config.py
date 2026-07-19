"""Unit tests for core.logging_config (issue #33).

``configure()`` sets up the package logger once with a single stderr handler;
level defaults to INFO, DEBUG when ``--debug`` is on ``sys.argv`` or
``IMAGE_ANNOTATOR_DEBUG`` is set in the environment. The package logger is a
process-global object, so the autouse fixture snapshots and restores its
handlers/level/propagate around every test — ordering never matters.
"""
import logging
import sys

import pytest

from digitalsreeni_image_annotator.core import logging_config

# Track whichever import root this test process loaded the package under
# (``configure()`` derives the same value from its own ``__name__``).
_PKG = logging_config._PKG


@pytest.fixture(autouse=True)
def _reset_package_logger():
    logger = logging.getLogger(_PKG)
    saved = (logger.handlers[:], logger.level, logger.propagate)
    logger.handlers = []
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    yield
    logger.handlers, level, propagate = saved
    logger.setLevel(level)
    logger.propagate = propagate


def _clean_env(monkeypatch):
    monkeypatch.delenv("IMAGE_ANNOTATOR_DEBUG", raising=False)
    monkeypatch.setattr(sys, "argv", ["prog"])


def test_configure_default_level_is_info(monkeypatch):
    _clean_env(monkeypatch)
    logger = logging_config.configure()
    assert logger.level == logging.INFO


def test_env_var_enables_debug(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("IMAGE_ANNOTATOR_DEBUG", "1")
    logger = logging_config.configure()
    assert logger.level == logging.DEBUG


def test_debug_flag_enables_debug(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["prog", "--debug"])
    logger = logging_config.configure()
    assert logger.level == logging.DEBUG


def test_configure_is_idempotent(monkeypatch):
    _clean_env(monkeypatch)
    logging_config.configure()
    logging_config.configure()
    logger = logging.getLogger(_PKG)
    assert len(logger.handlers) == 1


def test_get_logger_is_namespaced():
    child = logging_config.get_logger(_PKG + ".foo")
    assert child.parent is logging.getLogger(_PKG)


def test_pkg_root_derived_from_module_name():
    # Root is derived from __name__, not hardcoded, so it tracks the import
    # root the app was loaded under (installed vs. the `python -m src...` path).
    assert logging_config._PKG == logging_config.__name__.rsplit(".", 2)[0]


def test_child_logger_record_reaches_installed_handler(monkeypatch):
    """End-to-end: after configure(), an INFO record on a child logger actually
    reaches the installed handler. Guards the "_PKG must match the real import
    root" contract -- a hardcoded root silently drops every record under the
    ``python -m src....main`` launcher, where get_logger(__name__) is rooted at
    ``src.<pkg>`` rather than the configured tree."""
    _clean_env(monkeypatch)
    records = []
    root = logging_config.configure()
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    root.addHandler(handler)
    try:
        child = logging_config.get_logger(_PKG + ".sub.module")
        child.info("hello-e2e")
    finally:
        root.removeHandler(handler)
    assert any("hello-e2e" in m for m in records)
