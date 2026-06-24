"""Optional MLflow experiment tracking for model training (issue #74).

Wraps the optional ``mlflow`` dependency so the rest of the app can log
training runs without ever caring whether MLflow is installed or enabled.
When MLflow is missing **or** tracking is disabled, every method on
:class:`MLflowTracker` is a no-op and training behaves exactly as before
(graceful degradation).

``mlflow`` is imported lazily inside the methods that need it — never at
module top — mirroring the lazy-import idiom used for the other heavy,
optional libraries (``inference/sam_utils.py``, ``inference/dino_utils.py``)
so app startup stays fast and works without the extra installed.

Storage defaults to a local file store under ``<project>/mlruns`` (see
:func:`resolve_tracking_uri`), matching the issue's default.
"""

import os
import subprocess
import webbrowser

_DEFAULT_EXPERIMENT = "image-annotator-training"
_MLRUNS_DIRNAME = "mlruns"

# Cache the import probe — mlflow's import is non-trivial and the answer
# never changes within a process.
_AVAILABLE = None


def mlflow_available() -> bool:
    """Return True if the optional ``mlflow`` package can be imported."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            import mlflow  # noqa: F401
            _AVAILABLE = True
        except Exception:
            _AVAILABLE = False
    return _AVAILABLE


def resolve_tracking_uri(main_window=None) -> str:
    """Resolve the MLflow tracking store path.

    Precedence:
    1. A non-empty QSettings override (``tracking/mlflow_uri``).
    2. ``<current_project_dir>/mlruns`` when a project is open.
    3. ``<cwd>/mlruns`` as a last resort.

    Returns a filesystem path (MLflow accepts a local path as its tracking
    URI for the default file store).
    """
    from ..app_settings import load_mlflow_prefs

    _, uri, _ = load_mlflow_prefs()
    if uri:
        return uri
    project_dir = getattr(main_window, "current_project_dir", None)
    base = project_dir if project_dir else os.getcwd()
    return os.path.join(base, _MLRUNS_DIRNAME)


class MLflowTracker:
    """A graceful, optional MLflow run wrapper.

    Construct it (cheaply) on any thread, then call :meth:`start`,
    :meth:`log_metrics`, :meth:`log_artifact` and :meth:`end` **on the
    thread that does the training** — MLflow runs are thread-bound, so for
    SAM fine-tuning the run is started and ended inside the worker thread's
    ``train()`` call.

    If ``enabled`` is False or MLflow is not installed, every method is a
    no-op. Any error raised by MLflow during logging is caught and reported
    via ``log`` but never propagated — tracking must never abort training.
    """

    def __init__(
        self,
        enabled,
        tracking_uri,
        experiment_name=_DEFAULT_EXPERIMENT,
        run_name=None,
        log=None,
    ):
        self._want = bool(enabled)
        self._uri = tracking_uri
        self._experiment = experiment_name or _DEFAULT_EXPERIMENT
        self._run_name = run_name
        self._log = log
        self._active = False  # True only between a successful start() and end()

    # -- internal helpers ---------------------------------------------------

    def set_log(self, log):
        """Set the progress callback used for status lines.

        Callers pass a thread-safe sink (e.g. a Qt signal's ``emit``) so the
        tracker can report from a worker thread without touching the GUI
        directly.
        """
        self._log = log

    def _emit(self, msg):
        if self._log is not None:
            try:
                self._log(msg)
            except Exception:
                pass

    @property
    def active(self) -> bool:
        return self._active

    # -- public API ---------------------------------------------------------

    def start(self, params=None):
        """Open the run and log ``params``. Returns True if tracking is live."""
        if not self._want:
            return False
        if not mlflow_available():
            self._emit("MLflow not installed — skipping experiment tracking "
                       "(pip install 'digitalsreeni-image-annotator[tracking]').")
            return False
        try:
            import mlflow

            mlflow.set_tracking_uri(self._uri)
            mlflow.set_experiment(self._experiment)
            # Self-heal: a prior run stranded active (e.g. killed before end())
            # would make start_run raise "Run already active" and silently
            # degrade this run to untracked. Close it first.
            if mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow.start_run(run_name=self._run_name)
            if params:
                mlflow.log_params({k: v for k, v in params.items() if v is not None})
            self._active = True
            self._emit(f"MLflow tracking → {self._uri} (experiment "
                       f"'{self._experiment}').")
        except Exception as exc:  # never let tracking abort training
            self._active = False
            self._emit(f"MLflow tracking unavailable ({exc}); continuing untracked.")
        return self._active

    def log_metrics(self, metrics, step=None):
        if not self._active:
            return
        try:
            import mlflow

            for name, value in metrics.items():
                mlflow.log_metric(name, float(value), step=step)
        except Exception as exc:
            self._emit(f"MLflow metric logging failed ({exc}).")

    def log_artifact(self, path):
        if not self._active or not path:
            return
        try:
            import mlflow

            if os.path.exists(path):
                mlflow.log_artifact(path)
        except Exception as exc:
            self._emit(f"MLflow artifact logging failed ({exc}).")

    def end(self):
        if not self._active:
            return
        try:
            import mlflow

            mlflow.end_run()
        except Exception as exc:
            self._emit(f"MLflow run finalization failed ({exc}).")
        finally:
            self._active = False


def launch_mlflow_ui(tracking_uri, port=5000, log=None):
    """Launch the local ``mlflow ui`` server and open it in a browser.

    Returns ``(ok, message)``. ``ok`` is False (with a user-facing message)
    when MLflow is not installed, so the caller can show a dialog instead of
    crashing.
    """
    if not mlflow_available():
        return (
            False,
            "MLflow is not installed. Install it with:\n\n"
            "    pip install 'digitalsreeni-image-annotator[tracking]'\n\n"
            "or:  pip install mlflow",
        )
    try:
        subprocess.Popen(
            ["mlflow", "ui", "--backend-store-uri", str(tracking_uri),
             "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        webbrowser.open(f"http://localhost:{port}")
        msg = (f"Launching MLflow UI at http://localhost:{port}\n"
               f"Tracking store: {tracking_uri}\n\n"
               f"If the page doesn't load, port {port} may already be in use "
               f"by another MLflow server — check that tab.")
        if log is not None:
            log(msg)
        return True, msg
    except Exception as exc:
        return False, f"Could not start MLflow UI: {exc}"
