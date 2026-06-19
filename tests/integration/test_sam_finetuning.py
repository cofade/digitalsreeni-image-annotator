"""Tests for SAM 2 fine-tuning (issue bnsreenu#73).

Most tests are CI-friendly: pure geometry/loss/dataset helpers that need no
model weights or GPU. The full train→save→reload round trip is gated behind
``SAM_TRAIN_E2E=1`` and the presence of cached weights, since it needs the
~75 MB ``sam2_t.pt`` and is realistically GPU-only.
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest
from PyQt6.QtGui import QImage

from src.digitalsreeni_image_annotator.training.sam_trainer import (
    SampleGroup,
    bbox_to_mask,
    list_custom_models,
    make_custom_filename,
    mask_to_point,
    mask_to_xyxy,
    polygon_to_mask,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ── geometry helpers ─────────────────────────────────────────────────────────

class TestGeometry:
    def test_polygon_to_mask_fills_interior(self):
        mask = polygon_to_mask([10, 10, 40, 10, 40, 40, 10, 40], 50, 50)
        assert mask.dtype == bool and mask.shape == (50, 50)
        assert mask[25, 25] and not mask[5, 5]

    def test_bbox_to_mask(self):
        mask = bbox_to_mask([10, 10, 20, 20], 50, 50)  # x,y,w,h
        assert mask[15, 15] and not mask[45, 45]

    def test_mask_to_xyxy_tight(self):
        mask = np.zeros((50, 50), bool)
        mask[10:31, 5:26] = True
        x1, y1, x2, y2 = mask_to_xyxy(mask)
        assert (x1, y1, x2, y2) == (5, 10, 25, 30)

    def test_mask_to_xyxy_empty_is_none(self):
        assert mask_to_xyxy(np.zeros((10, 10), bool)) is None

    def test_mask_to_point_inside(self):
        mask = np.zeros((60, 60), bool)
        mask[20:41, 20:41] = True
        x, y = mask_to_point(mask)
        assert mask[int(y), int(x)]  # point lands inside the object

    def test_polygon_roundtrips_through_xyxy(self):
        mask = polygon_to_mask([10, 10, 40, 10, 40, 40, 10, 40], 50, 50)
        x1, y1, x2, y2 = mask_to_xyxy(mask)
        assert x1 >= 9 and y1 >= 9 and x2 <= 41 and y2 <= 41


# ── checkpoint naming (architecture-selection invariant) ─────────────────────

class TestCustomNaming:
    @pytest.mark.parametrize("base,token", [
        ("SAM 2 tiny", "sam2_t.pt"),
        ("SAM 2.1 base", "sam2.1_b.pt"),
        ("SAM 2.1 large", "sam2.1_l.pt"),
    ])
    def test_filename_keeps_base_token(self, base, token):
        # Ultralytics build_sam selects the architecture by ckpt.endswith(token);
        # a fine-tuned name MUST keep that suffix or SAM(path) can't load it.
        path = make_custom_filename(base, "my cool run!!")
        assert os.path.basename(path).endswith(token)

    def test_filename_sanitises_label(self):
        path = make_custom_filename("SAM 2 tiny", "a/b c:*?")
        name = os.path.basename(path)
        assert "/" not in name and ":" not in name and "*" not in name

    def test_filename_handles_empty_label(self):
        path = make_custom_filename("SAM 2 tiny", "")
        assert os.path.basename(path) == "finetuned_sam2_t.pt"

    def test_list_custom_models_returns_dict(self):
        assert isinstance(list_custom_models(), dict)


# ── SampleGroup lazy rasterisation ───────────────────────────────────────────

class TestSampleGroup:
    def test_load_rasterises_specs(self):
        img = np.zeros((50, 50, 3), np.uint8)
        specs = [
            {"segmentation": [10, 10, 40, 10, 40, 40, 10, 40]},
            {"bbox": [5, 5, 10, 10]},
        ]
        group = SampleGroup(lambda: img.copy(), specs)
        assert group.n_instances == 2
        out_img, instances = group.load()
        assert out_img.shape == (50, 50, 3)
        assert len(instances) == 2
        assert all(inst["mask"].dtype == bool for inst in instances)

    def test_load_drops_empty_masks(self):
        img = np.zeros((50, 50, 3), np.uint8)
        # Box entirely outside the image → no in-bounds pixels → dropped.
        group = SampleGroup(lambda: img.copy(), [{"bbox": [200, 200, 10, 10]}])
        _, instances = group.load()
        assert instances == []


# ── loss ─────────────────────────────────────────────────────────────────────

class TestLoss:
    def test_focal_dice_lower_when_correct(self):
        torch = pytest.importorskip("torch")
        from src.digitalsreeni_image_annotator.training.sam_trainer import _focal_dice_loss

        target = torch.zeros(1, 1, 16, 16)
        target[..., 4:12, 4:12] = 1.0
        good = (target * 12) - 6  # confident-correct logits
        bad = (1 - target) * 12 - 6  # confident-wrong logits
        l_good = _focal_dice_loss(good, target)
        l_bad = _focal_dice_loss(bad, target)
        assert torch.isfinite(l_good) and torch.isfinite(l_bad)
        assert float(l_good) < float(l_bad)


# ── dataset producers ────────────────────────────────────────────────────────

class TestDatasetProducers:
    def test_export_and_reload_folder_roundtrip(self, temp_dir):
        from src.digitalsreeni_image_annotator.io.export_formats import export_sam_dataset
        from src.digitalsreeni_image_annotator.training.sam_dataset import (
            build_groups_from_folder,
        )

        img = QImage(60, 60, QImage.Format.Format_RGB32)
        img.fill(0xFF202020)
        img_path = os.path.join(temp_dir, "img1.png")
        img.save(img_path)

        all_annotations = {
            "img1.png": {"cell": [{"segmentation": [10, 10, 40, 10, 40, 40, 10, 40]}]}
        }
        out_dir = os.path.join(temp_dir, "dataset")
        _, manifest_path = export_sam_dataset(
            all_annotations, {"cell": 1}, {"img1.png": img_path},
            slices=[], image_slices={}, output_dir=out_dir,
        )
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest["images"]) == 1

        groups = build_groups_from_folder(out_dir)
        assert len(groups) == 1
        _, instances = groups[0].load()
        assert len(instances) == 1

    def test_build_from_project_regular_image(self, temp_dir):
        from src.digitalsreeni_image_annotator.training.sam_dataset import (
            build_groups_from_project,
        )

        img = QImage(60, 60, QImage.Format.Format_RGB32)
        img.fill(0xFF808080)
        img_path = os.path.join(temp_dir, "img1.png")
        img.save(img_path)

        groups = build_groups_from_project(
            {"img1.png": {"cell": [{"bbox": [10, 10, 20, 20]}]}},
            {"img1.png": img_path}, slices=[], image_slices={},
        )
        assert len(groups) == 1
        image, instances = groups[0].load()
        assert image.shape[:2] == (60, 60) and len(instances) == 1

    def test_build_from_project_skips_unresolvable(self):
        from src.digitalsreeni_image_annotator.training.sam_dataset import (
            build_groups_from_project,
        )

        groups = build_groups_from_project(
            {"ghost.png": {"cell": [{"bbox": [1, 1, 5, 5]}]}},
            {}, slices=[], image_slices={},
        )
        assert groups == []


# ── ultralytics API-drift guard ──────────────────────────────────────────────

class TestUltralyticsAPI:
    def test_sam2predictor_exposes_forward_methods(self):
        """The native training loop reuses these SAM2Predictor methods; if an
        Ultralytics upgrade removes/renames them, fail loudly here rather than
        mid-training."""
        pytest.importorskip("ultralytics")
        from ultralytics.models.sam.predict import SAM2Predictor

        for name in ("get_im_features", "prompt_inference", "_inference_features",
                     "_prepare_prompts", "set_image"):
            assert hasattr(SAM2Predictor, name), f"SAM2Predictor.{name} missing"


# ── opt-in end-to-end (needs weights; realistically GPU) ─────────────────────

@pytest.mark.skipif(
    os.environ.get("SAM_TRAIN_E2E") != "1",
    reason="set SAM_TRAIN_E2E=1 to run the full train→save→reload test (needs weights)",
)
def test_end_to_end_train_save_reload(temp_dir):
    pytest.importorskip("ultralytics")
    from src.digitalsreeni_image_annotator.training.sam_trainer import (
        SAM_MODELS_DIR,
        SAMFineTuner,
    )

    if not os.path.exists(os.path.join(SAM_MODELS_DIR, "sam2_t.pt")):
        pytest.skip("sam2_t.pt not cached")

    def make(seed):
        rng = np.random.RandomState(seed)
        img = (rng.rand(120, 120, 3) * 255).astype(np.uint8)
        cy, cx = rng.randint(40, 80, 2)
        yy, xx = np.ogrid[:120, :120]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < 25 ** 2
        img[mask] = [220, 30, 30]
        x1, y1, x2, y2 = mask_to_xyxy(mask)
        return SampleGroup(
            lambda im=img: im.copy(),
            [{"bbox": [x1, y1, x2 - x1, y2 - y1]}],
        )

    out = os.path.join(temp_dir, "e2e_sam2_t.pt")
    res = SAMFineTuner().train(
        "SAM 2 tiny", [make(i) for i in range(2)],
        epochs=1, lr=1e-4, batch_size=1, freeze_image_encoder=True,
        prompt_type="bbox", out_path=out,
    )
    assert res["verified"] and os.path.exists(out)

    from ultralytics import SAM
    SAM(out)((np.random.rand(120, 120, 3) * 255).astype(np.uint8),
             bboxes=[[40, 40, 80, 80]], verbose=False)
