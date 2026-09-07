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
