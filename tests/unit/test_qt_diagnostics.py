"""Rules behind `sreeni-cli doctor` and the main.py Qt import guard (issue #92).

`diagnose` is a pure function over the mapping `qt_environment` produces, which is the
whole reason the two are separate: the failure being diagnosed needs a Windows box, a
Conda install and a *broken* Qt to reproduce, and none of those are available in CI.
Feeding it hand-built environments tests the rules on every platform.
"""

import os
import sys

import pytest

from src.digitalsreeni_image_annotator.core import qt_diagnostics as qd


def _dll(source, version, path=None):
    return {
        "path": path or os.path.join("C:\\", source, qd.QT_CORE_DLL),
        "source": source,
        "version": version,
    }


def _env(binding="6.11.0", runtime="6.11.1", sip="13.10.2",
         qt_core=None, msvc=None, conda_prefix=None):
    """An environment mapping shaped exactly like `qt_environment()` returns."""
    return {
        "platform": "win32",
        "python": "3.10.13",
        "executable": r"C:\envs\ann\python.exe",
        "prefix": r"C:\envs\ann",
        "conda_prefix": conda_prefix,
        "distributions": {
            qd.BINDING_DIST: binding,
            qd.RUNTIME_DIST: runtime,
            qd.SIP_DIST: sip,
        },
        "wheel_qt_bin": r"C:\envs\ann\Lib\site-packages\PyQt6\Qt6\bin",
        "qt_core_candidates": qt_core if qt_core is not None else [
            _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
        ],
        "msvc_candidates": msvc or [],
    }


def _severities(findings):
    return [finding.severity for finding in findings]


# --- the healthy case ------------------------------------------------------


def test_a_wheel_only_environment_is_clean():
    assert qd.diagnose(_env()) == []


def test_a_qt_after_the_wheel_is_not_a_problem():
    """System32 is searched last, so a Qt sitting there never wins and must not be
    reported -- a diagnostic that cries wolf on every healthy Windows box is worse
    than none."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
        _dll(qd.SOURCE_SYSTEM32, "6.2.0.0"),
    ])
    assert qd.diagnose(env) == []


# --- the reported failure --------------------------------------------------


def test_an_older_conda_qt_ahead_of_the_wheel_is_an_error():
    """The issue #92 scenario: conda-forge's Qt 6.8 shadows a PyQt6 6.11 wheel."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ], conda_prefix=r"C:\envs\ann")
    findings = qd.diagnose(env)

    assert _severities(findings) == ["error"]
    finding = findings[0]
    assert "6.11" in finding.detail and "6.8" in finding.detail
    assert qd.SOURCE_CONDA_LIBRARY_BIN in finding.detail
    # The remedy has to name a command, not just describe the problem.
    assert "python -m venv" in finding.remedy
    assert "PyQt6==6.8.*" in finding.remedy


def test_a_foreign_dll_wearing_the_name_gets_no_pip_pin_advice():
    """Whatever this is, it is not a Qt 6 build -- so `pip install "PyQt6==10.0.*"`
    would be a command that cannot resolve. Removing it is the only honest advice."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_APP_DIR, "10.0.26100.8875"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    findings = qd.diagnose(env)
    assert _severities(findings) == ["error"]
    assert "PyQt6==" not in findings[0].remedy
    assert "python -m venv" in findings[0].remedy


def test_the_environment_root_also_shadows_the_wheel():
    """For a Conda env the interpreter's own directory IS the env root, and Windows
    searches it before the directories PyQt6 registers for its bundled Qt."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_APP_DIR, "6.8.2.0"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    assert _severities(qd.diagnose(env)) == ["error"]


def test_a_matching_shadow_warns_but_does_not_error():
    """Same version, so nothing is broken today -- but the next PyQt6 upgrade breaks
    it, which is precisely how the reporter got there."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.11.1.0"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    findings = qd.diagnose(env)
    assert _severities(findings) == ["warning"]
    assert "upgrade" in findings[0].detail


def test_an_unreadable_shadow_version_still_warns():
    """No version resource means we cannot prove a mismatch, but a second Qt ahead of
    the wheel is still the thing worth telling the user about."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, None),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    findings = qd.diagnose(env)
    assert _severities(findings) == ["warning"]
    assert "unknown" in findings[0].detail


# --- pip-side skew, independent of Conda -----------------------------------


def test_missing_pyqt6_reports_that_and_stops():
    findings = qd.diagnose(_env(binding=None))
    assert _severities(findings) == ["error"]
    assert "not installed" in findings[0].title


def test_a_missing_qt_runtime_distribution_is_an_error():
    findings = qd.diagnose(_env(runtime=None))
    assert any(f.severity == "error" and qd.RUNTIME_DIST in f.title for f in findings)


def test_binding_and_runtime_minor_skew_is_an_error():
    findings = qd.diagnose(_env(binding="6.11.0", runtime="6.8.2"))
    assert any("different minor versions" in f.title for f in findings)


def test_a_patch_level_difference_is_normal():
    """PyQt6 6.11.0 ships against PyQt6-Qt6 6.11.1 -- flagging that would fire on
    every correct install."""
    assert qd.diagnose(_env(binding="6.11.0", runtime="6.11.1")) == []


def test_no_wheel_qt_means_no_shadowing_verdict():
    """A PyQt6 linked against a distro Qt (normal on Linux) ships no Qt6Core.dll of
    its own, so there is nothing to be shadowed and no finding to make."""
    env = _env(qt_core=[_dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0")])
    assert qd.diagnose(env) == []


# --- the second shadowing path: the MSVC runtime ---------------------------


def _msvc(source, version, name="msvcp140.dll"):
    return {
        "name": name,
        "path": os.path.join("C:\\", source, name),
        "source": source,
        "version": version,
    }


def test_an_older_local_msvc_runtime_warns():
    env = _env(msvc=[
        _msvc(qd.SOURCE_CONDA_LIBRARY_BIN, "14.29.30139.0"),
        _msvc(qd.SOURCE_SYSTEM32, "14.42.34433.0"),
    ])
    findings = qd.diagnose(env)
    assert _severities(findings) == ["warning"]
    assert "msvcp140.dll" in findings[0].title


def test_a_newer_or_equal_local_msvc_runtime_is_fine():
    env = _env(msvc=[
        _msvc(qd.SOURCE_CONDA_LIBRARY_BIN, "14.42.34433.0"),
        _msvc(qd.SOURCE_SYSTEM32, "14.42.34433.0"),
    ])
    assert qd.diagnose(env) == []


def test_msvc_without_a_system_copy_is_not_judged():
    """Nothing to compare against -- guessing would produce a finding on every
    non-Windows machine."""
    env = _env(msvc=[_msvc(qd.SOURCE_CONDA_LIBRARY_BIN, "14.29.30139.0")])
    assert qd.diagnose(env) == []


# --- version parsing -------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("6.11.1.0", (6, 11)),
    ("6.8.2", (6, 8)),
    ("6.11", (6, 11)),
    ("6", None),
    ("", None),
    (None, None),
    ("6.x.1", None),
])
def test_major_minor_parsing(value, expected):
    assert qd._major_minor(value) == expected


@pytest.mark.parametrize("value,expected", [
    ("14.42.34433.0", (14, 42, 34433, 0)),
    ("1.2", (1, 2)),
    ("1.2.beta", None),
    (None, None),
])
def test_version_tuple_parsing(value, expected):
    assert qd._version_tuple(value) == expected


# --- rendering -------------------------------------------------------------


def test_the_report_is_ascii_only():
    """A Windows console under a legacy code page mangles non-ASCII exactly when the
    output is redirected, which is what pasting a bug report does."""
    env = _env(qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    report = qd.format_report(env, qd.diagnose(env))
    report.encode("ascii")


def test_the_report_names_every_candidate_in_search_order():
    env = _env(qt_core=[
        _dll(qd.SOURCE_CONDA_LIBRARY_BIN, "6.8.2.0"),
        _dll(qd.SOURCE_WHEEL, "6.11.1.0"),
    ])
    report = qd.format_report(env, qd.diagnose(env))
    conda_at = report.index(qd.SOURCE_CONDA_LIBRARY_BIN)
    wheel_at = report.index(qd.SOURCE_WHEEL)
    assert conda_at < wheel_at
    assert "6.8.2.0" in report and "6.11.1.0" in report


def test_a_clean_report_says_so():
    report = qd.format_report(_env(), [])
    assert "No problems detected." in report


def test_import_failure_message_includes_the_original_error():
    message = qd.format_import_failure(
        ImportError("DLL load failed while importing QtCore: "
                    "The specified procedure could not be found.")
    )
    assert "DLL load failed" in message
    assert "sreeni-cli doctor" in message
    message.encode("ascii")


def test_import_failure_never_raises_even_if_the_probe_breaks(monkeypatch):
    """This runs inside an exception handler. A traceback out of the diagnostic would
    bury the very failure it exists to explain."""
    monkeypatch.setattr(
        qd, "qt_environment", lambda: (_ for _ in ()).throw(RuntimeError("probe blew up"))
    )
    message = qd.format_import_failure(ImportError("boom"))
    assert "boom" in message
    assert "diagnostic itself failed" in message


# --- the live probe --------------------------------------------------------


def test_qt_environment_runs_on_this_machine():
    """Shape check on the real environment: the keys diagnose() reads must exist, and
    the probe must not raise on any platform."""
    env = qd.qt_environment()
    for key in ("platform", "distributions", "qt_core_candidates", "msvc_candidates"):
        assert key in env
    assert isinstance(qd.diagnose(env), list)


def test_the_probe_does_not_import_pyqt6():
    """`find_spec` locates PyQt6 without executing it. If this module ever imported
    PyQt6 for real, it would crash in exactly the environment it exists to diagnose."""
    import subprocess

    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from digitalsreeni_image_annotator.core import qt_diagnostics as qd;"
        "qd.diagnose(qd.qt_environment());"
        "print('loaded' if 'PyQt6' in sys.modules else 'clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="PE version resources are Windows-only")
def test_dll_file_version_reads_a_real_system_dll():
    path = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                        "System32", "kernel32.dll")
    version = qd.dll_file_version(path)
    assert version is not None
    assert qd._version_tuple(version) is not None


def test_dll_file_version_returns_none_for_a_missing_file():
    assert qd.dll_file_version(os.path.join("C:\\", "nope", "absent.dll")) is None
