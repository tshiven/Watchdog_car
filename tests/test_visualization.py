"""Tests for vision/visualization.py, drawing onto synthetic frames."""

import numpy as np

from vision.visualization import (
    draw_detections,
    draw_error_vector,
    draw_frame_center,
    draw_hud,
    draw_target,
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def _blank_frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def _detection(bbox: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "class_id": 0,
        "class_name": "cat",
        "confidence": 0.87,
        "bbox": bbox,
        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        "width": x2 - x1,
        "height": y2 - y1,
        "area": (x2 - x1) * (y2 - y1),
    }


def test_draw_detections_marks_the_frame():
    frame = _blank_frame()

    draw_detections(frame, [_detection((100, 100, 200, 200))])

    assert frame.any()


def test_draw_detections_on_empty_list_leaves_frame_untouched():
    frame = _blank_frame()

    draw_detections(frame, [])

    assert not frame.any()


def test_draw_target_marks_the_frame_and_preserves_shape():
    frame = _blank_frame()

    result = draw_target(frame, (50, 50, 150, 150))

    assert result.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)
    assert result.any()


def test_draw_frame_center_draws_at_the_middle():
    frame = _blank_frame()

    draw_frame_center(frame)

    assert frame[FRAME_HEIGHT // 2, FRAME_WIDTH // 2].any()


def test_draw_error_vector_marks_the_frame():
    frame = _blank_frame()

    draw_error_vector(frame, (400, 300))

    assert frame.any()


def test_draw_hud_writes_each_line():
    single = _blank_frame()
    draw_hud(single, ["searching"])

    multiple = _blank_frame()
    draw_hud(multiple, ["searching", "dx=10 dy=-4"])

    assert single.any()
    assert multiple.sum() > single.sum()


def test_labels_near_the_top_edge_stay_inside_the_frame():
    frame = _blank_frame()

    # A box flush against the top would push its label off-frame if unclamped.
    draw_target(frame, (10, 0, 110, 100))

    assert frame.any()
