"""Central logging configuration for the application.

See the "Logging and Debug Output" section in
``docs/08_crosscutting_concepts.md`` and ADR-030 for the policy. The short
version: the whole package logs through one stdlib ``logging`` tree rooted at
``digitalsreeni_image_annotator``, configured once with a single stderr
handler. Level defaults to INFO; ``--debug`` on the command line or
``IMAGE_ANNOTATOR_DEBUG=1`` in the environment switches it to DEBUG.

Application code must use ``get_logger(__name__)`` and never a bare ``print``.
"""
import logging
import os
import sys

_PKG = "digitalsreeni_image_annotator"


def configure(level=None):
    """Configure the package logger once. Idempotent — safe to call twice
    (tests, re-entry): a second call must not add a second handler."""
    root = logging.getLogger(_PKG)
    if root.handlers:          # already configured
        return root
    if level is None:
        debug = ("--debug" in sys.argv
                 or os.environ.get("IMAGE_ANNOTATOR_DEBUG", "") not in ("", "0"))
        level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()   # stderr
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    return root


def get_logger(name):
    """Return a logger for ``name`` (pass ``__name__``). Because every module
    name starts with ``digitalsreeni_image_annotator.``, the returned logger
    inherits the package handler/level configured by :func:`configure`."""
    return logging.getLogger(name)
