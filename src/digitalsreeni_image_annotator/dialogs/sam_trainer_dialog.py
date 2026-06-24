"""Config dialog for SAM 2 fine-tuning.

Collects the hyperparameters for ``training.sam_trainer.SAMFineTuner.train``.
Progress is shown via the shared ``TrainingInfoDialog`` (reused from the YOLO
trainer) — this module only owns the *config* dialog.
"""

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from ..app_settings import load_mlflow_prefs
from ..inference.sam_utils import MODEL_NAMES


class SAMTrainConfigDialog(QDialog):
    """Modal config for a fine-tuning run. Read results via :meth:`get_config`."""

    def __init__(self, parent=None, *, default_name="my_finetune"):
        super().__init__(parent)
        self.setWindowTitle("Fine-Tune SAM Model")
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.base_model = QComboBox()
        self.base_model.addItems(MODEL_NAMES)
        # tiny/small are the realistic choices for desktop fine-tuning.
        form.addRow("Base model:", self.base_model)

        self.out_name = QLineEdit(default_name)
        form.addRow("Save as:", self.out_name)

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(10)
        form.addRow("Epochs:", self.epochs)

        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(1e-6, 1e-1)
        self.lr.setSingleStep(1e-5)
        self.lr.setValue(1e-4)
        form.addRow("Learning rate:", self.lr)

        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 64)
        self.batch_size.setValue(2)
        self.batch_size.setToolTip(
            "Gradient-accumulation count — the optimizer steps every N images "
            "(all of an image's objects are backpropagated together)."
        )
        form.addRow("Batch size:", self.batch_size)

        self.prompt_type = QComboBox()
        self.prompt_type.addItems(["bbox", "point"])
        self.prompt_type.setToolTip("Prompt derived from each ground-truth mask during training.")
        form.addRow("Train prompt:", self.prompt_type)

        self.train_encoder = QCheckBox("Also fine-tune image encoder (slower, needs more VRAM/data)")
        form.addRow("", self.train_encoder)

        # MLflow experiment tracking (issue #74). Defaults to the persisted
        # enable preference; the actual tracking URI / experiment name come
        # from Settings → Experiment Tracking.
        self.track_mlflow = QCheckBox("Track this run with MLflow")
        self.track_mlflow.setChecked(load_mlflow_prefs()[0])
        self.track_mlflow.setToolTip(
            "Log hyperparameters, per-epoch loss and the saved checkpoint to "
            "MLflow (configure the store under Settings → Experiment Tracking)."
        )
        form.addRow("", self.track_mlflow)

        layout.addLayout(form)

        note = QLabel(
            "Decoder-only (default) is fast and robust on modest data. "
            "Fine-tuning the image encoder can help on heavily domain-shifted "
            "imagery but needs more GPU memory and labels."
        )
        note.setWordWrap(True)
        # No inline color — inside a QDialog `palette(text)` resolves against the
        # stale OS palette and renders near-white in light mode. Let the global
        # QLabel stylesheet rule provide a theme-correct colour (No Hardcoded
        # Colors Rule, docs/08_crosscutting_concepts.md).
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> dict:
        return {
            "base_model": self.base_model.currentText(),
            "out_name": self.out_name.text().strip() or "my_finetune",
            "epochs": self.epochs.value(),
            "lr": self.lr.value(),
            "batch_size": self.batch_size.value(),
            "prompt_type": self.prompt_type.currentText(),
            "freeze_image_encoder": not self.train_encoder.isChecked(),
            "track_mlflow": self.track_mlflow.isChecked(),
        }
