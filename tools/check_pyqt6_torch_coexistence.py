"""
Phase 0 gate: confirm PyQt6 and PyTorch can coexist in one process.

Why this exists
---------------
The historical ADR-011 documented that on Windows + Python 3.14,
importing PyQt5 first and then loading PyTorch triggers
``WinError 1114`` (DLL load-order conflict between Qt's and Torch's
native deps). That motivated the now-deleted subprocess isolation
layer (sam_worker.py, dino_worker.py, check_worker_isolation.py).

Migrating to PyQt6 *should* eliminate the conflict — Qt6 reshuffled
its DLL packaging — but that is a hypothesis. This script is the
mechanical check. Run it before deleting any worker code.

The crucial bit: ``import PyQt6.QtCore`` alone does NOT load Qt's
native platform plugin (qwindows.dll on Windows, libqxcb on Linux).
The plugin is loaded lazily by ``QApplication.__init__``. That's
where the WinError 1114 actually triggers. So this script
constructs a ``QApplication`` after importing both PyQt6 and torch
to exercise the real interaction.

Usage
-----
    python tools/check_pyqt6_torch_coexistence.py

Run it especially on Windows + Python 3.14. Exit code 0 means the
combination loads cleanly *and* QApplication constructs without
crashing; exit code 1 means at least one stage failed.
"""

from __future__ import annotations

import platform
import sys
import traceback


def _try(label: str, fn) -> bool:
    print(f"[{label}] running ...", flush=True)
    try:
        result = fn()
    except BaseException:  # catch SystemExit / segfault recovery too
        print(f"[{label}] FAILED:")
        traceback.print_exc()
        return False
    if result is not None and hasattr(result, "__version__"):
        print(f"[{label}] OK — version: {result.__version__}", flush=True)
    else:
        print(f"[{label}] OK", flush=True)
    return True


def _construct_qapplication():
    """Force Qt's platform plugin to load.

    On Windows this is where qwindows.dll gets loaded, which is the
    site of the historical WinError 1114. We use 'offscreen' so the
    script runs in a headless CI / SSH context.
    """
    import os
    # Don't clobber an existing user setting — they may want to test
    # the real platform plugin specifically.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def main() -> int:
    print(f"Python:   {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine:  {platform.machine()}")
    print("-" * 60)

    # Order matters: PyQt first, then Torch, then Transformers.
    # This is the exact order the running app loads them in
    # (annotator_window imports PyQt at startup; torch is pulled
    # in by ultralytics/transformers when the user picks a model).
    ok = True
    ok &= _try("PyQt6.QtCore", lambda: __import__("PyQt6.QtCore", fromlist=["QtCore"]))
    ok &= _try("PyQt6.QtWidgets", lambda: __import__("PyQt6.QtWidgets", fromlist=["QtWidgets"]))
    ok &= _try("PyQt6.QtGui", lambda: __import__("PyQt6.QtGui", fromlist=["QtGui"]))
    ok &= _try("torch", lambda: __import__("torch"))
    ok &= _try("torchvision", lambda: __import__("torchvision"))
    ok &= _try("transformers", lambda: __import__("transformers"))
    ok &= _try("ultralytics", lambda: __import__("ultralytics"))
    # THIS is the real test — load the Qt platform plugin AFTER torch
    # is in the address space. Pure import_module above does not load
    # the platform plugin, so a green result without this step would
    # be a false positive.
    ok &= _try("QApplication construct (loads Qt platform plugin)", _construct_qapplication)

    print("-" * 60)
    if ok:
        print("RESULT: PyQt6 + Torch coexist cleanly, QApplication constructs.")
        print("        Subprocess removal unblocked.")
        return 0
    print("RESULT: at least one stage failed. Investigate before merging.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
