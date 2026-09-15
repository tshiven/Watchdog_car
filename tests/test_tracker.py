"""Tests for vision/tracker.py."""

from vision.tracker import Tracker, compute_center_error


def _detection(bbox: tuple[int, int, int, int], confidence: float = 0.9) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "class_id": 0,
        "class_name": "cat",
        "confidence": confidence,
        "bbox": bbox,
        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        "width": x2 - x1,
        "height": y2 - y1,
        "area": (x2 - x1) * (y2 - y1),
    }


def test_compute_center_error_target_at_frame_center():
    dx, dy = compute_center_error((320, 240), frame_width=640, frame_height=480)

    assert dx == 0
    assert dy == 0


def test_compute_center_error_target_offset():
    dx, dy = compute_center_error((400, 200), frame_width=640, frame_height=480)

    assert dx == 400 - 320
    assert dy == 200 - 240


def test_start_and_update_follows_moving_target():
    tracker = Tracker(frame_width=640, frame_height=480)
    tracker.start(_detection((100, 100, 200, 200)))

    result = tracker.update([_detection((110, 105, 210, 205))])

    assert tracker.is_tracking
    assert not result.lost
    assert result.bbox == (110, 105, 210, 205)
    assert result.dx == result.center[0] - 320
    assert result.dy == result.center[1] - 240


def test_follows_target_moving_most_of_its_own_width_per_frame():
    tracker = Tracker(frame_width=640, frame_height=480)
    tracker.start(_detection((100, 200, 200, 300)))

    # A 100px box crossing 60px per frame overlaps too little for an IoU gate,
    # but is unambiguously the same object.
    for bbox in [(160, 205, 260, 305), (220, 210, 320, 310), (280, 215, 380, 315)]:
        result = tracker.update([_detection(bbox)])

        assert not result.lost
        assert result.bbox == bbox


def test_picks_the_nearer_candidate_when_two_are_in_range():
    tracker = Tracker(frame_width=640, frame_height=480)
    tracker.start(_detection((100, 100, 200, 200)))

    near = _detection((115, 110, 215, 210))
    far = _detection((160, 160, 260, 260))
    result = tracker.update([far, near])

    assert result.bbox == near["bbox"]


def test_update_ignores_unrelated_detection_far_away():
    tracker = Tracker(frame_width=640, frame_height=480)
    tracker.start(_detection((100, 100, 200, 200)))

    # A detection nowhere near the tracked box shouldn't be adopted as a match.
    result = tracker.update([_detection((500, 400, 550, 450))])

    assert not result.lost
    assert result.bbox == (100, 100, 200, 200)


def test_track_lost_after_max_missed_frames():
    tracker = Tracker(frame_width=640, frame_height=480, max_missed_frames=2)
    tracker.start(_detection((100, 100, 200, 200)))

    unrelated = [_detection((500, 400, 550, 450))]
    tracker.update(unrelated)
    tracker.update(unrelated)
    result = tracker.update(unrelated)

    assert result.lost
    assert not tracker.is_tracking


def test_update_before_start_reports_lost():
    tracker = Tracker(frame_width=640, frame_height=480)

    result = tracker.update([_detection((0, 0, 10, 10))])

    assert result.lost
    assert result.bbox is None


def test_update_with_no_candidates_holds_last_position_until_timeout():
    tracker = Tracker(frame_width=640, frame_height=480, max_missed_frames=1)
    tracker.start(_detection((100, 100, 200, 200)))

    result = tracker.update([])

    assert not result.lost
    assert result.bbox == (100, 100, 200, 200)


def _small_fast_detection(x: int) -> dict:
    """A bottle-sized box at horizontal position `x`."""
    return _detection((x, 300, x + 60, 460), confidence=0.5)


def test_small_object_is_followed_as_well_as_a_large_one():
    # The bug this guards: the gate used to be a multiple of the target's own
    # size, so a 110px bottle was allowed 110px of movement per frame while a
    # 500px person was allowed 500px. Identical motion was followed for the
    # person and dropped for the bottle.
    tracker = Tracker(frame_width=1280, frame_height=720)
    tracker.start(_small_fast_detection(600))

    x = 600
    for _ in range(5):
        x += 140
        result = tracker.update([_small_fast_detection(x)])

        assert not result.lost
        assert result.bbox == (x, 300, x + 60, 460)


def test_person_sized_target_still_tracks():
    tracker = Tracker(frame_width=1280, frame_height=720)
    tracker.start(_detection((300, 60, 700, 660)))

    x = 300
    for _ in range(3):
        x += 140
        result = tracker.update([_detection((x, 60, x + 400, 660))])

        assert result.bbox == (x, 60, x + 400, 660)


def test_moving_target_is_re_acquired_after_a_detector_dropout():
    tracker = Tracker(frame_width=1280, frame_height=720)
    tracker.start(_small_fast_detection(600))

    x = 600
    for _ in range(3):
        x += 140
        tracker.update([_small_fast_detection(x)])

    for _ in range(3):  # the detector loses a small object for a few frames
        x += 140
        assert tracker.update([]).coasting

    x += 140
    result = tracker.update([_small_fast_detection(x)])

    assert not result.lost
    assert result.bbox == (x, 300, x + 60, 460)


def test_coasting_follows_the_targets_last_known_velocity():
    tracker = Tracker(frame_width=1280, frame_height=720)
    tracker.start(_small_fast_detection(600))
    for x in (740, 880, 1020):
        tracker.update([_small_fast_detection(x)])

    coasted = tracker.update([])

    assert coasted.coasting
    assert coasted.bbox[0] > 1020  # carried on, not frozen where it was


def test_coasting_is_reported_separately_from_a_confirmed_lock():
    tracker = Tracker(frame_width=640, frame_height=480)
    tracker.start(_detection((100, 100, 200, 200)))

    assert not tracker.update([_detection((110, 100, 210, 200))]).coasting
    assert tracker.update([]).coasting


def test_a_stationary_target_that_vanishes_is_still_declared_lost():
    # The gate opens along the target's velocity while it is missing. A target
    # that was not moving must gain nothing from that, or a vanished target
    # would start matching unrelated objects across the frame.
    tracker = Tracker(frame_width=640, frame_height=480, max_missed_frames=2)
    tracker.start(_detection((100, 100, 200, 200)))

    unrelated = [_detection((500, 400, 550, 450))]
    results = [tracker.update(unrelated) for _ in range(3)]

    assert [r.lost for r in results] == [False, False, True]


def test_a_box_of_a_wildly_different_size_is_not_adopted():
    tracker = Tracker(frame_width=640, frame_height=480)
    tracker.start(_detection((100, 100, 200, 200)))

    result = tracker.update([_detection((140, 140, 160, 160))])  # 5x smaller

    assert result.coasting
    assert result.bbox == (100, 100, 200, 200)


# --- Appearance matching ----------------------------------------------------
# These need real pixels, so they build frames rather than bare detection dicts.

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _bgr(hue: int, saturation: int = 200, value: int = 210) -> np.ndarray:
    return cv2.cvtColor(
        np.full((1, 1, 3), (hue, saturation, value), dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0]


def _two_bottle_frame(orange_x: int, blue_x: int) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[300:460, orange_x : orange_x + 60] = _bgr(15)
    frame[300:460, blue_x : blue_x + 60] = _bgr(110)
    return frame


def test_a_different_object_passing_closer_does_not_steal_the_lock():
    tracker = Tracker(frame_width=1280, frame_height=720)
    tracker.start(_small_fast_detection(600), _two_bottle_frame(600, 900))

    # The blue bottle cuts in nearer to the prediction than the orange one.
    frame = _two_bottle_frame(700, 660)
    result = tracker.update(
        [_small_fast_detection(700), _small_fast_detection(660)], frame
    )

    assert result.bbox == (700, 300, 760, 460)


def test_appearance_is_ignored_when_no_frame_is_supplied():
    # The matching logic has to stay usable from plain detection dicts.
    tracker = Tracker(frame_width=1280, frame_height=720)
    tracker.start(_small_fast_detection(600))

    assert not tracker.update([_small_fast_detection(700)]).lost


def test_target_survives_a_lighting_change():
    tracker = Tracker(frame_width=1280, frame_height=720)
    bright = np.zeros((720, 1280, 3), dtype=np.uint8)
    bright[300:460, 600:660] = _bgr(15, 200, 230)
    tracker.start(_small_fast_detection(600), bright)

    dim = np.zeros((720, 1280, 3), dtype=np.uint8)
    dim[300:460, 700:760] = _bgr(16, 150, 140)
    result = tracker.update([_small_fast_detection(700)], dim)

    assert not result.lost
    assert result.bbox == (700, 300, 760, 460)
