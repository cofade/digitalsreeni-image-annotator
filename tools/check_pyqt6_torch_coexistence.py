"""
Phase 0 gate: confirm PyQt6 and PyTorch can coexist in one process.

Why this exists
---------------
ADR-011 documents that on Windows + Python 3.14, importing PyQt5 first
and then loading PyTorch triggers `WinError 1114` (DLL load-order
conflict between Qt's and Torch's native deps). The whole subprocess
isolation layer (`sam_worker.py`, `dino_worker.py`,
`tools/check_worker_isolation.py`) exists to work around that bug.

The migration to PyQt6 *should* eliminate it — Qt6 reshuffled its
DLL packaging — but that is a hypothesis. This script is the
mechanical check. Run it before deleting any worker code.

Usage
-----
    python tools/check_pyqt6_torch_coexistence.py

Run it especially on Windows + Python 3.14. Exit code 0 means the
combination loads cleanly; exit code 1 means at least one import
failed (the failing module is printed). Linux/macOS runs are a
useful sanity check but were never the failure case.
"""

from __future__ import annotations

import platform
import sys
import traceback


def _try(label: str, import_fn) -> bool:
    print(f"[{label}] importing ...", flush=True)
    try:
        mod = import_fn()
    except Exception:
        print(f"[{label}] FAILED:")
        traceback.print_exc()
        return False
    version = getattr(mod, "__version__", "(no __version__ attr)")
    print(f"[{label}] OK — version: {version}", flush=True)
    return True


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

    print("-" * 60)
    if ok:
        print("RESULT: PyQt6 + Torch coexist cleanly. Subprocess removal unblocked.")
        return 0
    print("RESULT: at least one import failed. Keep the subprocess isolation.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
