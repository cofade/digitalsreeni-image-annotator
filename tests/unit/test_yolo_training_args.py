"""
Unit tests for the YOLO trainer forwarding the LR-schedule + early-stop knobs
to Ultralytics' ``model.train`` (issue #85).

`train_model` must pass through ``cos_lr`` / ``lr0`` / ``lrf`` / ``warmup_epochs``
/ ``patience``, deriving ``warmup_epochs`` as ~10% of epochs when not given. The
real ``model.train`` is replaced by a recorder so no training actually runs.
"""

from collections import deque

import pytest

from src.digitalsreeni_image_annotator.controllers.yolo_controller import (
    build_yolo_train_opts,
)
from src.digitalsreeni_image_annotator.dialogs import yolo_trainer as yt


def test_train_opts_on_is_warmup_cosine():
    opts = build_yolo_train_opts(50, cos_lr=True, lr0=0.01, patience=20)
    assert opts == {
        "cos_lr": True, "lr0": 0.01, "lrf": 0.1, "warmup_epochs": 5, "patience": 20,
    }


def test_train_opts_off_is_constant_lr():
    # cos_lr=False + lrf=1.0 (no decay) + warmup_epochs=0 ⇒ genuinely constant LR,
    # matching the SAM schedule toggle's "off" semantics.
    opts = build_yolo_train_opts(50, cos_lr=False, lr0=0.02, patience=0)
    assert opts["cos_lr"] is False
    assert opts["lrf"] == 1.0
    assert opts["warmup_epochs"] == 0
    assert opts["lr0"] == 0.02 and opts["patience"] == 0


class _FakeModel:
    def __init__(self):
        self.train_kwargs = None
        self.callbacks = {"on_train_epoch_end": [], "on_fit_epoch_end": []}

    def add_callback(self, name, cb):
        self.callbacks.setdefault(name, []).append(cb)

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        return "results"


def _make_trainer(tmp_path, monkeypatch):
    (tmp_path / "images" / "train").mkdir(parents=True)
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        "train: images/train\nval: images/train\nnc: 1\nnames: [a]\n", encoding="utf-8"
    )

    # Skip __init__ (needs a real main window); set just what train_model touches.
    trainer = yt.YOLOTrainer.__new__(yt.YOLOTrainer)
    trainer.project_dir = str(tmp_path)
    trainer.yaml_path = str(yaml_path)
    trainer.stop_training = False
    trainer.epoch_info = deque(maxlen=10)
    trainer._mlflow_url_emitted = False
    trainer.model = _FakeModel()

    monkeypatch.setattr(trainer, "_register_trained_model", lambda: None)
    # train_model resolves the device via ..core.torch_utils; pin it to CPU.
    from src.digitalsreeni_image_annotator.core import torch_utils
    monkeypatch.setattr(torch_utils, "resolve_torch_device", lambda: ("cpu", None))
    return trainer


def test_forwards_schedule_and_early_stop_args(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    trainer.train_model(epochs=50, imgsz=320, cos_lr=True, lr0=0.02, lrf=0.1, patience=15)

    kw = trainer.model.train_kwargs
    assert kw["epochs"] == 50 and kw["imgsz"] == 320
    assert kw["cos_lr"] is True
    assert kw["lr0"] == 0.02
    assert kw["lrf"] == 0.1
    assert kw["patience"] == 15
    assert kw["warmup_epochs"] == 5  # round(0.1 * 50)


@pytest.mark.parametrize("epochs,expected", [(50, 5), (3, 1), (10, 1)])
def test_warmup_epochs_defaults_to_ten_percent(tmp_path, monkeypatch, epochs, expected):
    trainer = _make_trainer(tmp_path, monkeypatch)
    trainer.train_model(epochs=epochs, imgsz=640)  # warmup_epochs not given
    assert trainer.model.train_kwargs["warmup_epochs"] == expected


def test_registers_both_progress_callbacks(tmp_path, monkeypatch):
    trainer = _make_trainer(tmp_path, monkeypatch)
    trainer.train_model(epochs=5, imgsz=640)
    # Cleared in the finally, so they end empty — but both keys must exist
    # (proving on_fit_epoch_end is wired alongside on_train_epoch_end).
    assert "on_train_epoch_end" in trainer.model.callbacks
    assert "on_fit_epoch_end" in trainer.model.callbacks
    assert trainer.model.callbacks["on_fit_epoch_end"] == []
