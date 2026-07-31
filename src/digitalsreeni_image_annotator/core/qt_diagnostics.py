"""Explain a failed PyQt6 / Qt import in terms the user can act on (issue #92, ADR-046).

Why this exists
---------------
Upgrading PyQt6 inside an *existing Conda* environment on Windows produces::

    ImportError: DLL load failed while importing QtCore:
    The specified procedure could not be found.

with Windows exception ``0xc0000139`` (``STATUS_ENTRYPOINT_NOT_FOUND``). That status
means a DLL imported a function name that the DLL the loader actually resolved does
not export -- i.e. **the wrong copy of a dependency won the search**. It is not a
defect in any particular PyQt6 release, which is why "downgrade PyQt6" appears to fix
it: the binding stops being newer than the Qt runtime it collided with.

Two shadowing paths are common in Conda environments, and this module looks for both:

1. ``qt6-main`` (pulled in by ``pyqt``, ``qtpy``, ``spyder``, ``napari``, matplotlib's
   Qt backend, ...) installs ``Qt6Core.dll`` into ``%CONDA_PREFIX%\\Library\\bin``.
   conda-forge's Qt lags PyPI's, so a newer PyQt6 binding binds to an older Qt.
2. Conda's ``vc14_runtime`` / ``vs2015_runtime`` ship ``msvcp140.dll`` into the
   environment. Newer Qt builds are compiled against a newer MSVC STL and import
   symbols an older ``msvcp140.dll`` does not export.

Constraints
-----------
**This module must never import PyQt6, at module level or inside a function.** It runs
*because* importing Qt failed, and it is reached from ``cli/``, which is bound by the
Qt-free rule (ADR-041) and covered by the subprocess test in
``tests/integration/test_cli.py``.

It also never *loads* a DLL. :func:`dll_file_version` reads the PE version resource out
of the file, so it is safe to call on the very DLL that is crashing the process.

All output is ASCII only, matching the convention documented in ``cli/main.py``: a
Windows console under a legacy code page turns non-ASCII into mojibake exactly when the
output is redirected, which is what a bug report does.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

QT_CORE_DLL = "Qt6Core.dll"
MSVC_RUNTIME_DLLS = ("msvcp140.dll", "vcruntime140.dll")

# Distributions that together make up a working PyQt6: the bindings, the Qt runtime
# they are compiled against, and the sip runtime. A skew between the first two is its
# own failure mode, independent of anything Conda does.
BINDING_DIST = "PyQt6"
RUNTIME_DIST = "PyQt6-Qt6"
SIP_DIST = "PyQt6-sip"

# Where the loader looks, labelled for the report. Ordering matters and is the whole
# point of the diagnosis -- see _qt_core_candidates for the rationale behind it.
SOURCE_APP_DIR = "python-dir"
SOURCE_CONDA_LIBRARY_BIN = "conda-library-bin"
SOURCE_WHEEL = "pyqt6-wheel"
SOURCE_SYSTEM32 = "system32"


@dataclass(frozen=True)
class Finding:
    """One diagnosed problem. ``severity`` is ``"error"`` or ``"warning"``."""

    severity: str
    title: str
    detail: str
    remedy: str


# --- primitives ------------------------------------------------------------


def _distribution_version(name: str) -> str | None:
    """Installed version of ``name``, read from package *metadata*.

    Deliberately not ``import PyQt6; PyQt6.QtCore.PYQT_VERSION_STR`` -- the import is
    the thing that just failed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - corrupt metadata on a user's machine
        return None


def wheel_qt_bin_dir() -> str | None:
    """Directory holding the Qt DLLs that the PyQt6 wheel ships, or ``None``.

    ``find_spec`` locates the package without executing it, so this stays safe in an
    environment where importing PyQt6 crashes the interpreter.
    """
    from importlib.util import find_spec

    try:
        spec = find_spec("PyQt6")
    except Exception:  # pragma: no cover - broken meta path finder
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    candidate = os.path.join(list(spec.submodule_search_locations)[0], "Qt6", "bin")
    return candidate if os.path.isdir(candidate) else None


def dll_file_version(path: str) -> str | None:
    """Four-part file version of a DLL, read from its PE version resource.

    Reads the file; never loads it as a module. Returns ``None`` off Windows, when the
    file carries no version resource, or on any failure -- a diagnostic that raises
    while diagnosing a crash is worse than one that says "unknown".
    """
    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    class VSFixedFileInfo(ctypes.Structure):
        _fields_ = [
            ("dwSignature", wintypes.DWORD),
            ("dwStrucVersion", wintypes.DWORD),
            ("dwFileVersionMS", wintypes.DWORD),
            ("dwFileVersionLS", wintypes.DWORD),
            ("dwProductVersionMS", wintypes.DWORD),
            ("dwProductVersionLS", wintypes.DWORD),
            ("dwFileFlagsMask", wintypes.DWORD),
            ("dwFileFlags", wintypes.DWORD),
            ("dwFileOS", wintypes.DWORD),
            ("dwFileType", wintypes.DWORD),
            ("dwFileSubtype", wintypes.DWORD),
            ("dwFileDateMS", wintypes.DWORD),
            ("dwFileDateLS", wintypes.DWORD),
        ]

    try:
        version_dll = ctypes.WinDLL("version")
        get_size = version_dll.GetFileVersionInfoSizeW
        get_size.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        get_size.restype = wintypes.DWORD
        size = get_size(path, None)
        if not size:
            return None

        get_info = version_dll.GetFileVersionInfoW
        get_info.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p
        ]
        get_info.restype = wintypes.BOOL
        buffer = ctypes.create_string_buffer(size)
        if not get_info(path, 0, size, buffer):
            return None

        query = version_dll.VerQueryValueW
        query.argtypes = [
            ctypes.c_void_p, wintypes.LPCWSTR,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
        ]
        query.restype = wintypes.BOOL
        block = ctypes.c_void_p()
        length = ctypes.c_uint()
        if not query(buffer, "\\", ctypes.byref(block), ctypes.byref(length)):
            return None

        info = ctypes.cast(block, ctypes.POINTER(VSFixedFileInfo)).contents
        if info.dwSignature != 0xFEEF04BD:
            return None
        high, low = info.dwFileVersionMS, info.dwFileVersionLS
        return (
            f"{high >> 16}.{high & 0xFFFF}.{low >> 16}.{low & 0xFFFF}"
        )
    except Exception:  # pragma: no cover - any Win32 failure means "unknown"
        return None


def _major_minor(version: str | None) -> tuple[int, int] | None:
    """``"6.11.1.0"`` -> ``(6, 11)``. ``None`` when it cannot be parsed."""
    if not version:
        return None
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


# --- environment capture ---------------------------------------------------


def _conda_prefix() -> str | None:
    """The active Conda prefix, or ``None`` outside Conda.

    ``CONDA_PREFIX`` is only set by an activated shell, so a ``conda-meta`` directory
    next to the interpreter is checked too -- the app is often launched from a
    shortcut or an IDE that never ran ``conda activate``.
    """
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix and os.path.isdir(prefix):
        return prefix
    if os.path.isdir(os.path.join(sys.prefix, "conda-meta")):
        return sys.prefix
    return None


def _qt_core_candidates(conda_prefix: str | None, wheel_bin: str | None) -> list[dict[str, Any]]:
    """Every ``Qt6Core.dll`` the loader could pick, in the order it would pick them.

    The order is the diagnosis. For a dependency of an extension module Windows uses
    ``LOAD_LIBRARY_SEARCH_DEFAULT_DIRS``: the **application directory** (which for an
    installed interpreter is the directory holding ``python.exe``, and for Conda *is*
    the environment root) comes before the user directories that PyQt6 registers via
    ``os.add_dll_directory`` for its own bundled Qt, which come before ``System32``.
    So a stray Qt in the environment root or in Conda's ``Library\\bin`` wins over the
    one the wheel ships, and the binding then resolves against the wrong runtime.

    ``PATH`` is deliberately not probed: since Python 3.8 it is no longer consulted
    when resolving an extension module's dependencies, so reporting it would send
    people to edit a variable that cannot be the cause.
    """
    searched: list[tuple[str, str]] = [(os.path.dirname(sys.executable), SOURCE_APP_DIR)]
    if conda_prefix:
        searched.append((os.path.join(conda_prefix, "Library", "bin"), SOURCE_CONDA_LIBRARY_BIN))
    if wheel_bin:
        searched.append((wheel_bin, SOURCE_WHEEL))
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    searched.append((system32, SOURCE_SYSTEM32))

    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for directory, source in searched:
        if not directory:
            continue
        path = os.path.join(directory, QT_CORE_DLL)
        key = os.path.normcase(os.path.abspath(path))
        if key in seen or not os.path.isfile(path):
            continue
        seen.add(key)
        found.append({"path": path, "source": source, "version": dll_file_version(path)})
    return found


def _msvc_candidates(conda_prefix: str | None) -> list[dict[str, Any]]:
    """MSVC runtime copies inside the environment, paired with the System32 one."""
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    directories: list[tuple[str, str]] = [(os.path.dirname(sys.executable), SOURCE_APP_DIR)]
    if conda_prefix:
        directories.append(
            (os.path.join(conda_prefix, "Library", "bin"), SOURCE_CONDA_LIBRARY_BIN)
        )
    directories.append((system32, SOURCE_SYSTEM32))

    found: list[dict[str, Any]] = []
    for name in MSVC_RUNTIME_DLLS:
        for directory, source in directories:
            if not directory:
                continue
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            found.append(
                {"name": name, "path": path, "source": source,
                 "version": dll_file_version(path)}
            )
    return found


def qt_environment() -> dict[str, Any]:
    """Everything :func:`diagnose` needs, gathered from the live environment.

    Split from :func:`diagnose` so the rules are a pure function over plain data and
    can be tested without a Conda install, a Windows box, or a broken Qt.
    """
    conda_prefix = _conda_prefix()
    wheel_bin = wheel_qt_bin_dir()
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "prefix": sys.prefix,
        "conda_prefix": conda_prefix,
        "distributions": {
            BINDING_DIST: _distribution_version(BINDING_DIST),
            RUNTIME_DIST: _distribution_version(RUNTIME_DIST),
            SIP_DIST: _distribution_version(SIP_DIST),
        },
        "wheel_qt_bin": wheel_bin,
        "qt_core_candidates": _qt_core_candidates(conda_prefix, wheel_bin),
        "msvc_candidates": _msvc_candidates(conda_prefix),
    }


# --- the rules -------------------------------------------------------------


_CLEAN_ENV_REMEDY = (
    "Install into a clean virtual environment that has no Qt of its own:\n"
    "    python -m venv .venv\n"
    "    .venv\\Scripts\\activate\n"
    "    pip install -e ."
)


def diagnose(env: dict[str, Any]) -> list[Finding]:
    """Rules over the :func:`qt_environment` mapping. Pure; no I/O."""
    findings: list[Finding] = []
    distributions = env.get("distributions", {})
    binding = distributions.get(BINDING_DIST)
    runtime = distributions.get(RUNTIME_DIST)

    if binding is None:
        findings.append(Finding(
            severity="error",
            title="PyQt6 is not installed",
            detail="No PyQt6 distribution metadata was found for this interpreter.",
            remedy="pip install -e .   (in the environment you actually run the app from)",
        ))
        return findings

    # A binding/runtime skew is its own bug and is not Conda's doing: it happens when
    # only one of the two pip distributions gets upgraded.
    binding_mm, runtime_mm = _major_minor(binding), _major_minor(runtime)
    if runtime is None:
        findings.append(Finding(
            severity="error",
            title=f"{RUNTIME_DIST} is missing",
            detail=f"{BINDING_DIST} {binding} is installed but its Qt runtime is not.",
            remedy=f'pip install --force-reinstall "{BINDING_DIST}=={binding}"',
        ))
    elif binding_mm and runtime_mm and binding_mm != runtime_mm:
        findings.append(Finding(
            severity="error",
            title="PyQt6 and its Qt runtime are different minor versions",
            detail=f"{BINDING_DIST} {binding} against {RUNTIME_DIST} {runtime}.",
            remedy=f'pip install --force-reinstall "{BINDING_DIST}=={binding}"',
        ))

    findings.extend(_diagnose_qt_shadowing(env, runtime))
    findings.extend(_diagnose_msvc_shadowing(env))
    return findings


def _diagnose_qt_shadowing(env: dict[str, Any], runtime: str | None) -> list[Finding]:
    """Flag a ``Qt6Core.dll`` the loader would reach before the wheel's own."""
    candidates = env.get("qt_core_candidates", [])
    wheel_index = next(
        (i for i, c in enumerate(candidates) if c["source"] == SOURCE_WHEEL), None
    )
    # Nothing shipped by the wheel means nothing to shadow -- either PyQt6 links a
    # system Qt on purpose (Linux distro packages do) or the install is incomplete,
    # which the distribution rules above already cover.
    if wheel_index is None:
        return []

    expected = _major_minor(runtime)
    findings: list[Finding] = []
    for candidate in candidates[:wheel_index]:
        found = _major_minor(candidate["version"])
        shadow = (
            f"{candidate['path']} (version {candidate['version'] or 'unknown'}) "
            f"is searched before {env.get('wheel_qt_bin')}."
        )
        if expected and found and expected != found:
            remedy = (
                f"{_CLEAN_ENV_REMEDY}\n"
                "  or remove the conflicting Qt if nothing in the environment needs it:\n"
                "    conda remove qt6-main"
            )
            # Only offer "downgrade the binding to match" when the shadowing file is
            # actually a Qt of the same major. Anything else is a foreign DLL wearing
            # the name, and emitting `pip install "PyQt6==10.0.*"` would hand the user
            # a command that cannot resolve.
            if found[0] == expected[0]:
                remedy += (
                    "\n  or match the binding to the Qt already present:\n"
                    f'    pip install "PyQt6=={found[0]}.{found[1]}.*"'
                )
            findings.append(Finding(
                severity="error",
                title="A different Qt runtime shadows the one PyQt6 ships",
                detail=(
                    f"{shadow} PyQt6 expects Qt {expected[0]}.{expected[1]}.x but that copy "
                    f"is {found[0]}.{found[1]}.x, so QtCore resolves against a Qt that does "
                    "not export the symbols it needs -- the 0xc0000139 crash."
                ),
                remedy=remedy,
            ))
        else:
            findings.append(Finding(
                severity="warning",
                title="Another Qt runtime is on the search path",
                detail=(
                    f"{shadow} Its version matches, so it is not currently breaking the "
                    "import, but a PyQt6 upgrade would make the two diverge."
                ),
                remedy=_CLEAN_ENV_REMEDY,
            ))
    return findings


def _diagnose_msvc_shadowing(env: dict[str, Any]) -> list[Finding]:
    """Flag an environment-local MSVC runtime older than the system's.

    Newer Qt builds import STL symbols that an older ``msvcp140.dll`` does not export,
    which produces the same ``0xc0000139`` from a completely different DLL.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for candidate in env.get("msvc_candidates", []):
        by_name.setdefault(candidate["name"], []).append(candidate)

    findings: list[Finding] = []
    for name, candidates in by_name.items():
        system = next((c for c in candidates if c["source"] == SOURCE_SYSTEM32), None)
        local = [c for c in candidates if c["source"] != SOURCE_SYSTEM32]
        if system is None or not local:
            continue
        system_version = _version_tuple(system["version"])
        for candidate in local:
            local_version = _version_tuple(candidate["version"])
            if not system_version or not local_version or local_version >= system_version:
                continue
            findings.append(Finding(
                severity="warning",
                title=f"An older {name} shadows the system one",
                detail=(
                    f"{candidate['path']} is version {candidate['version']} while "
                    f"{system['path']} is {system['version']}. Qt is built against the "
                    "newer MSVC runtime and imports symbols the older copy lacks."
                ),
                remedy=(
                    "Update the environment's runtime (conda update vc14_runtime) or "
                    f"delete {candidate['path']} so the system copy is used."
                ),
            ))
    return findings


def _version_tuple(version: str | None) -> tuple[int, ...] | None:
    """Full dotted version as a comparable tuple, or ``None`` if unparsable."""
    if not version:
        return None
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return None


# --- rendering -------------------------------------------------------------


def format_report(env: dict[str, Any], findings: list[Finding]) -> str:
    """Full environment report for ``sreeni-cli doctor``."""
    lines: list[str] = ["Qt environment", "--------------"]
    lines.append(f"  platform     {env.get('platform')}")
    lines.append(f"  python       {env.get('python')}")
    lines.append(f"  interpreter  {env.get('executable')}")
    conda_prefix = env.get("conda_prefix")
    lines.append(f"  conda        {conda_prefix or 'not detected'}")

    lines.append("")
    lines.append("Distributions")
    for name, value in (env.get("distributions") or {}).items():
        lines.append(f"  {name:<12} {value or 'not installed'}")

    candidates = env.get("qt_core_candidates") or []
    lines.append("")
    lines.append(f"{QT_CORE_DLL} on the loader search path (first match wins)")
    if candidates:
        for index, candidate in enumerate(candidates, start=1):
            version = candidate.get("version") or "unknown"
            lines.append(f"  {index}. [{candidate['source']}] {version}")
            lines.append(f"     {candidate['path']}")
    else:
        lines.append("  none found")

    lines.append("")
    if not findings:
        lines.append("No problems detected.")
        return "\n".join(lines)

    lines.append(f"Findings ({len(findings)})")
    lines.append("--------")
    lines.extend(_format_findings(findings))
    return "\n".join(lines)


def _format_findings(findings: list[Finding]) -> list[str]:
    lines: list[str] = []
    for index, finding in enumerate(findings, start=1):
        lines.append(f"{index}. [{finding.severity}] {finding.title}")
        lines.append(f"   {finding.detail}")
        for remedy_line in finding.remedy.splitlines():
            lines.append(f"   {remedy_line}")
        lines.append("")
    return lines


def format_import_failure(exc: BaseException) -> str:
    """Message shown when the GUI entry point cannot import Qt.

    Never raises: this runs inside an exception handler, and a traceback from the
    diagnostic would bury the original failure it is meant to explain.
    """
    lines = [
        "ERROR: the Qt runtime (PyQt6) failed to load, so the application cannot start.",
        "",
        f"  {type(exc).__name__}: {exc}",
        "",
    ]
    try:
        env = qt_environment()
        findings = diagnose(env)
        if findings:
            lines.append("Likely cause")
            lines.append("------------")
            lines.extend(_format_findings(findings))
        else:
            lines.append(
                "No conflicting Qt installation was detected, so this is not the known"
            )
            lines.append("Conda shadowing problem. Full environment detail:")
            lines.append("")
            lines.append(format_report(env, findings))
            lines.append("")
    except Exception:  # pragma: no cover - the diagnostic must never mask the error
        lines.append("(the environment diagnostic itself failed to run)")
        lines.append("")

    lines.append("Run 'sreeni-cli doctor' for the full report -- it does not load Qt, so it")
    lines.append("still works in an environment where this application cannot start.")
    lines.append("See the Troubleshooting section of README.md.")
    return "\n".join(lines)
