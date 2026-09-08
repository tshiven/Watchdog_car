"""Tests for scripts/run_watchdog.py, driving the pipeline with a fake detector."""

import numpy as np

from scripts.run_watchdog import (
    STATUS_LOCKED,
    STATUS_LOST,
    STATUS_SEARCHING,
    crop_to_bbox,
    process_frame,
)
from vision.tracker import Tracker

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ORANGE_BGR = (0, 140, 255)
GRAY_BGR = (128, 128, 128)


class FakeDetector:
    """Returns a canned list of detections, ignoring the frame."""

    def __init__(self, *frames_of_detections: list[dict]) -> None:
        self._frames = list(frames_of_detections)

    def detect(self, frame, target_class=None) -> list[dict]:
        return self._frames.pop(0) if self._frames else []


def _detection(bbox, class_name="cat", confidence=0.9) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "class_id": 15,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        "width": x2 - x1,
        "height": y2 - y1,
        "area": (x2 - x1) * (y2 - y1),
    }


def _blank_frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def _tracker() -> Tracker:
    return Tracker(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)


def test_no_detections_reports_searching():
    outcome = process_frame(_blank_frame(), FakeDetector([]), _tracker(), "cat", None)

    assert outcome.status == STATUS_SEARCHING
    assert outcome.bbox is None


def test_matching_detection_locks_on_and_reports_center_error():
    detection = _detection((100, 100, 200, 200))

    outcome = process_frame(_blank_frame(), FakeDetector([detection]), _tracker(), "cat", None)

    assert outcome.status == STATUS_LOCKED
    assert outcome.bbox == (100, 100, 200, 200)
    assert outcome.dx == 150 - FRAME_WIDTH // 2
    assert outcome.dy == 150 - FRAME_HEIGHT // 2


def test_detection_of_another_class_is_ignored():
    outcome = process_frame(
        _blank_frame(), FakeDetector([_detection((100, 100, 200, 200), "dog")]), _tracker(), "cat", None
    )

    assert outcome.status == STATUS_SEARCHING


def test_low_confidence_detection_is_not_locked_onto():
    weak = _detection((100, 100, 200, 200), confidence=0.2)

    outcome = process_frame(_blank_frame(), FakeDetector([weak]), _tracker(), "cat", None)

    assert outcome.status == STATUS_SEARCHING


def test_requested_color_decides_between_two_candidates():
    frame = _blank_frame()
    frame[100:200, 100:200] = ORANGE_BGR
    frame[100:200, 400:500] = GRAY_BGR
    orange_bbox = (100, 100, 200, 200)
    gray_bbox = (400, 100, 500, 200)
    detector = FakeDetector([_detection(gray_bbox), _detection(orange_bbox)])

    outcome = process_frame(frame, detector, _tracker(), "cat", "orange")

    assert outcome.status == STATUS_LOCKED
    assert outcome.bbox == orange_bbox


def test_target_is_followed_across_frames():
    tracker = _tracker()
    detector = FakeDetector(
        [_detection((100, 100, 200, 200))],
        [_detection((160, 105, 260, 205))],
    )
    frame = _blank_frame()

    first = process_frame(frame, detector, tracker, "cat", None)
    second = process_frame(frame, detector, tracker, "cat", None)

    assert first.status == STATUS_LOCKED
    assert second.status == STATUS_LOCKED
    assert second.bbox == (160, 105, 260, 205)
    assert second.dx > first.dx


def test_target_is_declared_lost_after_it_disappears():
    tracker = _tracker()
    frame = _blank_frame()
    process_frame(frame, FakeDetector([_detection((100, 100, 200, 200))]), tracker, "cat", None)

    statuses = [
        process_frame(frame, FakeDetector([]), tracker, "cat", None).status for _ in range(10)
    ]

    assert STATUS_LOST in statuses
    assert statuses[-1] == STATUS_SEARCHING  # falls back to searching afterwards


def test_crop_to_bbox_clips_a_box_that_runs_past_the_frame_edge():
    frame = _blank_frame()

    crop = crop_to_bbox(frame, (-50, -50, 100, 100))

    assert crop.shape[0] == 100
    assert crop.shape[1] == 100
