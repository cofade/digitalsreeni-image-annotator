"""
Grounding DINO utilities --- delegates to an isolated subprocess.

On Windows + Python 3.14, loading PyTorch after PyQt5 causes
WinError 1114. Running DINO in a clean subprocess avoids the issue.
"""

import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from PyQt5.QtGui import QImage

from .sam_utils import _qimage_to_numpy
from .utils import models_base_dir


GDINO_MODEL_NAMES = [
    "grounding-dino-base",
    "grounding-dino-tiny",
]


def _gdino_local_path(model_name: str) -> str:
    """Canonical local install path for a Grounding DINO model."""
    return os.path.join(models_base_dir(), model_name)


# Kept for backwards compatibility / external callers. Computed lazily via
# the helper so it always agrees with sam_worker / annotator_window.
GDINO_MODEL_PATHS = {
    "grounding-dino-base": _gdino_local_path("grounding-dino-base"),
    "grounding-dino-tiny": _gdino_local_path("grounding-dino-tiny"),
}

GDINO_REPO_IDS = {
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
}


class DINOUtils:
    """Thin wrapper that forwards DINO work to a subprocess worker."""

    def __init__(self):
        self._worker_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "dino_worker.py"
        )

    def _send_request(self, request: dict) -> dict:
        """Spawn the DINO worker, send JSON, and return parsed response."""
        env = os.environ.copy()
        for possible in ("VIRTUAL_ENV", "CONDA_PREFIX"):
            v = os.environ.get(possible)
            if v:
                env[possible] = v
                break
        # Force the worker to write UTF-8 so cp1252 (Windows) doesn't choke
        # on non-ASCII bytes from torch/transformers warnings.
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.run(
            [sys.executable, self._worker_script],
            input=json.dumps(request) + "\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        if proc.returncode != 0:
            err_text = proc.stderr.strip() if proc.stderr else "(no stderr)"
            raise RuntimeError(
                f"DINO worker exited with code {proc.returncode}.\nstderr: {err_text}"
            )

        # Echo worker stdout (includes device diagnostics) to parent console
        lines = (proc.stdout or "").strip().splitlines()
        for line in lines[:-1]:
            print(line)

        try:
            return json.loads(lines[-1])
        except (json.JSONDecodeError, IndexError):
            out_text = proc.stdout.strip() if proc.stdout else "(no stdout)"
            raise RuntimeError(
                f"DINO worker returned non-JSON output.\nstdout: {out_text}"
            )

    @staticmethod
    def _save_image_temp(image: QImage) -> str:
        """Convert QImage to a temporary file and return the path."""
        arr = _qimage_to_numpy(image)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        from PIL import Image as PILImage
        PILImage.fromarray(arr).save(tmp.name)
        tmp.close()
        return tmp.name

    def detect(self, image, class_configs, model_name="grounding-dino-base",
               custom_model_path=None):
        """
        Run text-prompted detection.

        Parameters
        ----------
        image : QImage
            The image to detect objects in.
        class_configs : list[dict]
            Each dict: {"name": str, "phrases": [str], "box_thr": float,
                        "txt_thr": float, "nms_thr": float}
        model_name : str
            One of GDINO_MODEL_NAMES, or "custom".
        custom_model_path : str | None
            Local path for custom/fine-tuned model.

        Returns
        -------
        list[dict] | None
            Each dict: {"class_name", "bbox": [x1,y1,x2,y2], "score", "label"}
            Returns None on error.
        """
        model_path = custom_model_path
        if model_path is None:
            model_path = GDINO_MODEL_PATHS.get(model_name)
        if model_path is None:
            print(f"Unknown DINO model: {model_name}")
            return None

        # Convert relative paths to absolute from project root
        if not os.path.isabs(model_path):
            # Try relative to the package directory, then cwd
            abs_from_pkg = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                model_path
            )
            if os.path.exists(abs_from_pkg):
                model_path = abs_from_pkg
            else:
                abs_from_cwd = os.path.join(os.getcwd(), model_path)
                if os.path.exists(abs_from_cwd):
                    model_path = abs_from_cwd

        tmp_path = None
        try:
            tmp_path = self._save_image_temp(image)
            request = {
                "action": "detect",
                "image_path": tmp_path,
                "class_configs": class_configs,
                "model_path": model_path,
            }
            result = self._send_request(request)
        except Exception:
            traceback.print_exc()
            return None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if "error" in result:
            print(f"DINO worker error: {result['error']}")
            return None

        return result.get("results", [])

    def download_model(self, model_name: str):
        """
        Download model from Hugging Face Hub into the canonical local path.
        Returns the absolute local path on success, or None on error.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("huggingface_hub not installed. Cannot download models.")
            return None

        repo_id = GDINO_REPO_IDS.get(model_name)
        if not repo_id:
            print(f"No repo ID for model: {model_name}")
            return None

        local_path = GDINO_MODEL_PATHS.get(model_name) or _gdino_local_path(model_name)
        if os.path.exists(local_path):
            print(f"Model already exists at {local_path}")
            return local_path

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {repo_id} -> {local_path} ...")
        snapshot_download(repo_id, local_dir=local_path)
        print(f"Done. Model at {local_path}")
        return local_path
