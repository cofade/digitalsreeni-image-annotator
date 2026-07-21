"""Unit tests for the VideoTimeline widget (issue #48).

Widget-only: exercises ``widgets/video_timeline.py`` with no main window. The
critical invariants are (1) ``set_current_frame`` NEVER re-emits
``frameSelected`` (a feedback loop back into ``switch_slice``), (2) genuine user
interaction DOES emit it exactly once, and (3) the fps guard keeps MM:SS finite.
"""

import pytest

from digitalsreeni_image_annotator.widgets.video_timeline import VideoTimeline


@pytest.fixture
def timeline(qt_application):
    tl = VideoTimeline()
    yield tl
    tl.deleteLater()


def test_set_video_configures_slider_range(timeline):
    timeline.set_video(100, 25.0)
    assert timeline.slider.minimum() == 0
    assert timeline.slider.maximum() == 99


def test_set_video_clamp_does_not_emit(qt_application):
    # THE feedback-loop invariant: set_video reconfigures the range, and if a
    # smaller total clamps a non-zero slider value Qt fires valueChanged. The
    # _updating guard must swallow it, or reconfiguring the timeline on an
    # annotation mutation would spuriously emit frameSelected -> switch_slice
    # -> jump to frame 0.
    from digitalsreeni_image_annotator.widgets.video_timeline import VideoTimeline

    tl = VideoTimeline()
    tl.set_video(100, 25.0)
    tl.set_current_frame(80)          # slider now at 80
    emitted = []
    tl.frameSelected.connect(emitted.append)
    tl.set_video(10, 25.0)            # setMaximum(9) clamps 80 -> 9 (valueChanged)
    assert emitted == []              # guard swallowed the clamp emission
    tl.deleteLater()


def test_set_current_frame_does_not_emit(timeline):
    """Programmatic sync (from switch_slice) must NOT emit frameSelected —
    otherwise it re-enters switch_slice (feedback loop)."""
    timeline.set_video(100, 25.0)
    emitted = []
    timeline.frameSelected.connect(emitted.append)

    timeline.set_current_frame(50)

    assert emitted == []
    assert timeline.slider.value() == 50


def test_user_slider_move_emits_once(timeline):
    """A user-driven slider change (not programmatic, not a live drag) emits
    frameSelected exactly once."""
    timeline.set_video(100, 25.0)
    emitted = []
    timeline.frameSelected.connect(emitted.append)

    # _updating is False and the handle isn't held down → valueChanged is a
    # genuine user move.
    timeline.slider.setValue(50)

    assert emitted == [50]


def test_label_shows_mmss_at_frame(timeline):
    """Frame 50 @ 25 fps is 2.0 s → 00:02 appears in the position label."""
    timeline.set_video(100, 25.0)
    timeline.set_current_frame(50)
    assert "00:02" in timeline.label.text()


def test_set_annotated_frames_stores(timeline):
    timeline.set_video(100, 25.0)
    timeline.set_annotated_frames({0, 99})
    assert timeline.annotated_frames == {0, 99}


def test_set_video_resets_annotated_marks(timeline):
    timeline.set_video(100, 25.0)
    timeline.set_annotated_frames({3, 4})
    timeline.set_video(50, 30.0)  # reconfigure → marks reset
    assert timeline.annotated_frames == set()


def test_zero_fps_does_not_divide_by_zero(timeline):
    """A 0/NaN fps container must not ZeroDivisionError; fps is guarded to 30."""
    timeline.set_video(10, 0)
    timeline.set_current_frame(5)  # _fmt_time would divide by zero without guard
    assert timeline.label.text()  # non-empty, finite MM:SS


def test_clear_resets_and_emits_nothing(timeline):
    timeline.set_video(100, 25.0)
    timeline.set_annotated_frames({1, 2})
    emitted = []
    timeline.frameSelected.connect(emitted.append)

    timeline.clear()

    assert emitted == []
    assert timeline.annotated_frames == set()
    assert timeline.slider.maximum() == 0
    assert timeline.label.text() == ""
