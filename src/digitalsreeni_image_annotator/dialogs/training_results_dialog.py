"""Post-training results panel (issue #74).

Replaces the `Training complete` message box, which said nothing about whether
the model was any good and left the user to set up their own prediction run to
find out.

Shows what the trainer actually reported, links to the MLflow run, and offers
one button that closes the loop: run the fresh model on the current image, into
the existing review overlay.

Metrics that a given training path does not report are **omitted, not shown as
placeholders**. An empty row reads as "the model scored nothing", which is a
very different claim from "this path does not report that".
"""

import webbrowser

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..core import model_sidecar


class TrainingResultsDialog(QDialog):
    def __init__(self, main_window, summary, registry):
        super().__init__(main_window)
        self.mw = main_window
        self.summary = summary
        self.registry = registry
        self.setWindowTitle("Training complete")

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"The {summary['model_type'].upper()} model is trained and is "
                "now the active prediction model — no separate load step."
            )
        )

        metrics_box = QGroupBox("Results")
        metrics_form = QFormLayout(metrics_box)
        rows = model_sidecar.format_metrics(summary.get("metrics"))
        if rows:
            for label, value in rows:
                metrics_form.addRow(label, QLabel(value))
        else:
            metrics_form.addRow(
                QLabel("This training path reported no metrics.")
            )
        layout.addWidget(metrics_box)

        files_box = QGroupBox("Saved")
        files_form = QFormLayout(files_box)
        weights = QLabel(str(summary.get("weights_path") or "—"))
        weights.setWordWrap(True)
        files_form.addRow("Weights", weights)
        if summary.get("sidecar_path"):
            sidecar = QLabel(str(summary["sidecar_path"]))
            sidecar.setWordWrap(True)
            files_form.addRow("Sidecar", sidecar)
        size_mb = registry.models_dir_size_mb()
        if size_mb is not None:
            # Every run leaves a checkpoint. Showing the number is the honest
            # alternative to quietly deleting old ones.
            files_form.addRow(
                "Models folder", QLabel(f"{size_mb:.1f} MB (all runs kept)")
            )
        layout.addWidget(files_box)

        buttons_row = QHBoxLayout()
        self.try_button = QPushButton("Try it on the current image")
        self.try_button.setToolTip(
            "Run the fresh model on the image you have open and review the "
            "predictions."
        )
        self.try_button.setEnabled(registry.can_try_now())
        if not registry.can_try_now():
            self.try_button.setToolTip("Open an image first.")
        self.try_button.clicked.connect(self._try_now)
        buttons_row.addWidget(self.try_button)

        self.mlflow_button = QPushButton("Open in MLflow")
        url = summary.get("mlflow_url")
        # The MLflow URL arrives asynchronously (ADR-027), so it may not exist
        # yet when this panel opens. Disabled rather than absent, so the
        # capability is discoverable.
        self.mlflow_button.setEnabled(bool(url))
        if not url:
            self.mlflow_button.setToolTip("The MLflow run link is not available yet.")
        self.mlflow_button.clicked.connect(
            lambda: webbrowser.open(summary["mlflow_url"])
        )
        buttons_row.addWidget(self.mlflow_button)
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.accept)
        layout.addWidget(close_box)

    def set_mlflow_url(self, url):
        """Enable the MLflow button once the URL arrives."""
        self.summary["mlflow_url"] = url
        self.mlflow_button.setEnabled(bool(url))

    def _try_now(self):
        self.registry.try_on_current_image()
        self.accept()
