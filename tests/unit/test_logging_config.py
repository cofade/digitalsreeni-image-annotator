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

_PKG = "digitalsreeni_image_annotator"


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
