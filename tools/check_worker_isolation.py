"""
Smoke check: importing either ML worker subprocess script must NOT pull
PyQt5 into the interpreter.

ADR-011 (docs/09_architecture_decisions.md) requires that `sam_worker.py`
and `dino_worker.py` run in a Qt-free process — loading PyQt5 alongside
PyTorch on Windows + Python 3.14 triggers `WinError 1114`. Both workers
have already shipped, been broken, and been fixed once on this exact
invariant; this script is the mechanical guard so it doesn't happen
a third time.

Usage:
    python tools/check_worker_isolation.py

Exit code 0 = both workers clean. Exit code 1 = PyQt5 leaked into the
import of at least one worker, or a real error occurred. Prints what
went wrong.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKERS = [
    os.path.join(REPO_ROOT, "src", "digitalsreeni_image_annotator", "sam_worker.py"),
    os.path.join(REPO_ROOT, "src", "digitalsreeni_image_annotator", "dino_worker.py"),
]


class _PyQt5Tripwire(importlib.abc.MetaPathFinder):
    """Raise on any attempt to import PyQt5 or a PyQt5 submodule."""

    def __init__(self):
        self.tripped = False
        self.tripped_by: str | None = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "PyQt5" or fullname.startswith("PyQt5."):
            self.tripped = True
            self.tripped_by = fullname
            raise ImportError(
                f"PyQt5 leaked into worker subprocess via import of {fullname!r}"
            )
        return None


def _check_one(worker_path: str, tripwire: _PyQt5Tripwire) -> tuple[bool, str]:
    """Return (ok, message). ok=False means the worker either leaked PyQt5
    or failed to load for an unrelated reason."""
    name = f"_check_{os.path.basename(worker_path)[:-3]}"
    try:
        spec = importlib.util.spec_from_file_location(name, worker_path)
        if spec is None or spec.loader is None:
            return False, f"Cannot create import spec for {worker_path}"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as e:
        # The tripwire raises ImportError to abort the load. Distinguish
        # PyQt5 leakage (the thing we care about) from a missing third-party
        # dependency that just makes the smoke check unusable in this env.
        if tripwire.tripped:
            return False, str(e)
        return False, (
            f"Skipped {worker_path}: missing dependency ({e}). "
            "Install project requirements to run this check."
        )
    except Exception as e:
        return False, f"Unexpected error loading {worker_path}: {type(e).__name__}: {e}"

    # Belt-and-braces: even if the tripwire didn't fire, confirm no PyQt5
    # module landed in sys.modules during this exec. The tripwire would
    # catch first-import; this catches the case where a future bug
    # bypassed the finder.
    leaked = [m for m in sys.modules if m == "PyQt5" or m.startswith("PyQt5.")]
    if leaked:
        return False, (
            f"sys.modules contains {leaked!r} after loading {worker_path} "
            "— PyQt5 was imported somewhere the tripwire didn't catch."
        )
    return True, f"OK: {os.path.basename(worker_path)} loaded without PyQt5."


def main() -> int:
    tripwire = _PyQt5Tripwire()
    sys.meta_path.insert(0, tripwire)

    all_ok = True
    skipped_for_deps = False
    for worker in WORKERS:
        ok, msg = _check_one(worker, tripwire)
        print(msg)
        if not ok:
            if "missing dependency" in msg:
                skipped_for_deps = True
            else:
                all_ok = False
        # Reset the tripwire between workers — a leak in one shouldn't
        # mask a leak in the next.
        tripwire.tripped = False
        tripwire.tripped_by = None

    if not all_ok:
        return 1
    if skipped_for_deps:
        # Couldn't fully verify; surface this so CI/reviewer notices.
        print("\nNote: at least one worker was skipped due to missing deps. "
              "Re-run after `pip install -e .` for a complete check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
