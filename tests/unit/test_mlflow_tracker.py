"""Unit tests for the optional MLflow tracker (issue #74).

The point of these tests is the graceful-degradation contract: with tracking
disabled, or with mlflow not installed, the tracker is an inert no-op that
never raises and never pretends to be active. The live-logging path is
exercised only when mlflow is actually importable (skipped otherwise).
"""

import os

import pytest

from digitalsreeni_image_annotator.training import mlflow_tracker
from digitalsreeni_image_annotator.training.mlflow_tracker import (
    MLflowTracker,
    resolve_tracking_uri,
)


class TestDisabledTracker:
    def test_all_methods_are_noops_and_inactive(self):
        msgs = []
        t = MLflowTracker(enabled=False, tracking_uri="ignored", log=msgs.append)
        assert t.start({"a": 1}) is False
        assert t.active is False
        # None of these raise, even though no run is open.
        t.log_metrics({"loss": 0.5}, step=1)
        t.log_artifact("/nonexistent/path")
        t.end()
        assert msgs == []  # disabled stays silent

    def test_enabled_but_mlflow_missing_degrades(self, monkeypatch):
        # Force the "not installed" branch regardless of the real environment.
        monkeypatch.setattr(mlflow_tracker, "_AVAILABLE", False)
        msgs = []
        t = MLflowTracker(enabled=True, tracking_uri="x", log=msgs.append)
        assert t.start({"a": 1}) is False
        assert t.active is False
        t.log_metrics({"loss": 1.0})  # safe no-op
        t.end()
        assert any("MLflow not installed" in m for m in msgs)


class TestMlflowAvailable:
    def test_probe_is_cached(self, monkeypatch):
        monkeypatch.setattr(mlflow_tracker, "_AVAILABLE", None)
        first = mlflow_tracker.mlflow_available()
        # Second call must return the cached value without re-probing.
        assert mlflow_tracker.mlflow_available() is first


class _FakeMW:
    def __init__(self, project_dir=None):
        if project_dir is not None:
            self.current_project_dir = project_dir


class TestResolveTrackingUri:
    def test_override_wins(self, monkeypatch, tmp_path):
        # load_mlflow_prefs is imported inside the function, so patch the source.
        monkeypatch.setattr(
            "digitalsreeni_image_annotator.app_settings.load_mlflow_prefs",
            lambda settings=None: (True, str(tmp_path / "override"), "exp"),
        )
        assert resolve_tracking_uri(_FakeMW(project_dir="/proj")) == str(
            tmp_path / "override"
        )

    def test_project_dir_used_when_no_override(self, monkeypatch):
        monkeypatch.setattr(
            "digitalsreeni_image_annotator.app_settings.load_mlflow_prefs",
            lambda settings=None: (True, "", "exp"),
        )
        uri = resolve_tracking_uri(_FakeMW(project_dir="/proj"))
        assert uri == os.path.join("/proj", "mlruns")

    def test_cwd_fallback(self, monkeypatch):
        monkeypatch.setattr(
            "digitalsreeni_image_annotator.app_settings.load_mlflow_prefs",
            lambda settings=None: (True, "", "exp"),
        )
        uri = resolve_tracking_uri(_FakeMW(project_dir=None))
        assert uri == os.path.join(os.getcwd(), "mlruns")


class TestLiveLogging:
    """Exercised only when mlflow is actually installed."""

    def test_run_records_params_and_metrics(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mlflow_tracker, "_AVAILABLE", None)
        if not mlflow_tracker.mlflow_available():
            pytest.skip("mlflow not installed")
        import mlflow

        store = str(tmp_path / "mlruns")
        t = MLflowTracker(
            enabled=True, tracking_uri=store, experiment_name="ut-exp",
            run_name="ut-run",
        )
        assert t.start({"epochs": 3, "lr": 1e-4, "skip": None}) is True
        t.log_metrics({"loss": 0.9}, step=1)
        t.log_metrics({"loss": 0.4}, step=2)
        t.end()
        assert t.active is False

        mlflow.set_tracking_uri(store)
        exp = mlflow.get_experiment_by_name("ut-exp")
        assert exp is not None
        runs = mlflow.search_runs([exp.experiment_id])
        assert len(runs) == 1
        assert runs.iloc[0]["params.epochs"] == "3"
        # None-valued params are filtered out.
        assert "params.skip" not in runs.columns or runs.iloc[0].get(
            "params.skip"
        ) is None
        assert runs.iloc[0]["metrics.loss"] == 0.4
