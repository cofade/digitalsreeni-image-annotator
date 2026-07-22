"""Unit tests for SAM 3 video object tracking (issue #51, ADR-038).

The real ~3.45 GB gated ``SAM3VideoPredictor`` is never constructed. These
exercise the REAL ``SAM3Utils._track_blocking`` loop (the monkeypatch seam) by
injecting a FAKE predictor via ``ultralytics.models.sam.SAM3VideoPredictor`` —
so ``_run_sync`` serialisation, the ``_mask_to_polygon`` conversion, ``None``
(object-absent) handling and the ``should_cancel`` early-stop are all covered
with deterministic per-frame data.
"""

import numpy as np
import pytest

from digitalsreeni_image_annotator.inference.sam3_utils import SAM3Utils


# --- fake per-frame result objects (duck-typed to Ultralytics Results) ------

class _FakeArr:
    """Mimics a torch tensor: ``.cpu().numpy()`` yields the wrapped ndarray."""

    def __init__(self, arr):
        self._arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeMasks:
    def __init__(self, arr):
        self.data = _FakeArr(arr)


class _FakeBoxes:
    def __init__(self, conf):
        self.conf = _FakeArr(np.array(conf, dtype=float))


class _FakeResult:
    """One per-frame video result. ``mask=None`` → object absent this frame.

    Real Ultralytics ``masks.data`` is shape ``(N, H, W)`` — one plane per
    detected object — so a 2D mask is promoted to a single-object stack.
    """

    def __init__(self, mask=None, conf=None):
        if mask is None:
            self.masks = None
        else:
            arr = np.asarray(mask)
            if arr.ndim == 2:
                arr = arr[None, ...]  # (H, W) → (1, H, W)
            self.masks = _FakeMasks(arr)
        self.boxes = _FakeBoxes(conf) if conf is not None else None


def _square_mask(size=20, box=(5, 5, 15, 15)):
    """A filled square → a valid contour (area > 10, ≥ 6 flat coords)."""
    m = np.zeros((size, size), dtype=np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 1
    return m


def _install_fake_predictor(monkeypatch, frames):
    """Route ``_track_blocking``'s lazy import at a fake predictor that streams
    ``frames`` — the real SAM3VideoPredictor is never touched."""

    class _FakePredictor:
        def __init__(self, overrides=None):
            self.overrides = overrides

        def __call__(self, source=None, bboxes=None, stream=False):
            return list(frames)

    monkeypatch.setattr(
        "ultralytics.models.sam.SAM3VideoPredictor", _FakePredictor, raising=False
    )


def _loaded_utils():
    u = SAM3Utils()
    # Mark loaded WITHOUT constructing the real predictor (track() gates on it).
    u.loaded = True
    u._predictor = object()
    u._device = "cpu"
    return u


# --- tests ------------------------------------------------------------------

def test_track_returns_empty_when_not_loaded(qt_application):
    u = SAM3Utils()  # loaded is False
    assert u.track("clip.avi", 0, [1, 2, 3, 4]) == []


def test_track_maps_results_and_handles_absent(
    qt_application, monkeypatch, make_test_video, tmp_path
):
    # Frames straddle a threshold: high / low / absent / high.
    frames = [
        _FakeResult(_square_mask(), conf=[0.9]),
        _FakeResult(_square_mask(), conf=[0.4]),
        _FakeResult(mask=None),                     # object absent
        _FakeResult(_square_mask(), conf=[0.95]),
    ]
    _install_fake_predictor(monkeypatch, frames)
    # _track_blocking reads the real clip to slice frames; a 4-frame video +
    # direction="forward" maps the fake stream 1:1 onto indices 0..3.
    video = make_test_video(tmp_path, name="clip.avi", frames=4)
    u = _loaded_utils()

    out = u.track(video, 0, [1, 1, 15, 15], direction="forward")

    # One (frame_idx, result) per streamed frame, in order.
    assert [idx for idx, _ in out] == [0, 1, 2, 3]
    # High/low frames carry a polygon + their score; absent frame is None.
    assert out[0][1]["score"] == pytest.approx(0.9)
    assert len(out[0][1]["segmentation"]) >= 6
    assert out[1][1]["score"] == pytest.approx(0.4)
    assert out[2][1] is None
    assert out[3][1]["score"] == pytest.approx(0.95)


def test_track_should_cancel_stops_early(
    qt_application, monkeypatch, make_test_video, tmp_path
):
    frames = [
        _FakeResult(_square_mask(), conf=[0.9]),
        _FakeResult(_square_mask(), conf=[0.8]),
        _FakeResult(_square_mask(), conf=[0.7]),
        _FakeResult(_square_mask(), conf=[0.6]),
    ]
    _install_fake_predictor(monkeypatch, frames)
    video = make_test_video(tmp_path, name="clip.avi", frames=4)
    u = _loaded_utils()

    calls = {"n": 0}

    def cancel():
        # Cancel is polled at the START of each streamed frame; return True on
        # the 2nd poll so exactly one frame is accumulated before the break.
        calls["n"] += 1
        return calls["n"] >= 2

    out = u.track(video, 0, [1, 1, 15, 15], direction="forward", should_cancel=cancel)

    assert len(out) == 1
    assert out[0][0] == 0
    # The fake genuinely saw the cancel signal (polled at least twice).
    assert calls["n"] >= 2


def test_track_absent_when_no_conf_defaults_score_one(
    qt_application, monkeypatch, make_test_video, tmp_path
):
    # A mask but no boxes/conf → score defaults to 1.0 (present, confident).
    frames = [_FakeResult(_square_mask(), conf=None)]
    _install_fake_predictor(monkeypatch, frames)
    video = make_test_video(tmp_path, name="clip.avi", frames=2)
    u = _loaded_utils()

    out = u.track(video, 0, [1, 1, 15, 15], direction="forward")

    assert out[0][1]["score"] == pytest.approx(1.0)


def test_track_arbitrary_seed_bidirectional(
    qt_application, monkeypatch, make_test_video, tmp_path
):
    """Seed on a NON-zero frame with direction='both': the forward run covers
    seed..end and the backward run covers seed..0, so every frame is mapped and
    the shared seed frame de-dupes. Guards the temp-video slice/re-map wiring
    (#51 arbitrary-frame seed + bidirectional)."""
    # Fake streams 3 frames per run (forward [2,3,4], backward [2,1,0]).
    frames = [_FakeResult(_square_mask(), conf=[0.9]) for _ in range(3)]
    _install_fake_predictor(monkeypatch, frames)
    video = make_test_video(tmp_path, name="clip.avi", frames=5)
    u = _loaded_utils()

    out = u.track(video, 2, [1, 1, 15, 15], direction="both")

    # forward {2,3,4} + backward {2,1,0}, seed 2 de-duped -> all 5 frames sorted.
    assert [idx for idx, _ in out] == [0, 1, 2, 3, 4]
    assert all(r is not None for _, r in out)
