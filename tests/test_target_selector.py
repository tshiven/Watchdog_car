"""Tests for vision/target_selector.py."""

from vision.target_selector import select_target


def make_detection(
    class_name: str,
    confidence: float,
    area: int,
) -> dict:
    """Create a fake detection matching the shared detection contract."""
    width = int(area**0.5)
    height = width

    return {
        "class_id": 0,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": (0, 0, width, height),
        "center": (width // 2, height // 2),
        "width": width,
        "height": height,
        "area": area,
    }


def test_wrong_class_is_rejected():
    detections = [
        make_detection("dog", 0.95, 1000),
    ]

    result = select_target(detections, "cat")

    assert result is None


def test_low_confidence_is_rejected():
    detections = [
        make_detection("cat", 0.2, 1000),
    ]

    result = select_target(detections, "cat")

    assert result is None


def test_best_candidate_is_selected():
    first_cat = make_detection("cat", 0.6, 1000)
    second_cat = make_detection("cat", 0.9, 1000)

    detections = [first_cat, second_cat]

    result = select_target(detections, "cat")

    assert result is second_cat


def test_best_candidate_does_not_have_to_be_first():
    weaker_cat = make_detection("cat", 0.6, 1000)
    better_cat = make_detection("cat", 0.95, 1000)

    detections = [weaker_cat, better_cat]

    result = select_target(detections, "cat")

    assert result is better_cat


def test_color_score_can_change_winner():
    first_cat = make_detection("cat", 0.95, 1000)
    second_cat = make_detection("cat", 0.80, 1000)

    detections = [first_cat, second_cat]
    color_scores = [0.1, 1.0]

    result = select_target(
        detections,
        "cat",
        requested_color="orange",
        color_scores=color_scores,
    )

    assert result is second_cat


def test_no_color_works_without_color_scores():
    first_cat = make_detection("cat", 0.7, 1000)
    second_cat = make_detection("cat", 0.9, 1000)

    result = select_target(
        [first_cat, second_cat],
        "cat",
    )

    assert result is second_cat


def test_larger_object_gets_size_bonus():
    smaller_cat = make_detection("cat", 0.8, 1000)
    larger_cat = make_detection("cat", 0.8, 4000)

    result = select_target(
        [smaller_cat, larger_cat],
        "cat",
    )

    assert result is larger_cat


def test_empty_detection_list_returns_none():
    result = select_target([], "cat")

    assert result is None


def test_color_scores_must_match_detection_count():
    detections = [
        make_detection("cat", 0.8, 1000),
        make_detection("cat", 0.9, 1000),
    ]

    try:
        select_target(
            detections,
            "cat",
            requested_color="orange",
            color_scores=[0.5],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched color scores")

def test_mid_confidence_small_object_can_be_locked_onto():
    # yolov8n scores a person ~0.94 but a bottle ~0.42 and a phone ~0.38. The
    # floor used to be 0.50, so those two were drawn on screen every frame and
    # could never be selected -- the system searched for what it was showing.
    for class_name, confidence in [("bottle", 0.42), ("cell phone", 0.38)]:
        detections = [make_detection(class_name, confidence, 1000)]

        assert select_target(detections, class_name) is not None


def test_confidence_below_the_detectors_own_floor_is_still_rejected():
    assert select_target([make_detection("cat", 0.2, 1000)], "cat") is None


def test_class_matching_is_case_insensitive():
    # Class names come from the model's table; a typed request should not have
    # to match their casing.
    detections = [make_detection("Cell Phone", 0.8, 1000)]

    assert select_target(detections, "cell phone") is detections[0]


def test_color_outranks_confidence_and_size():
    # Colour is the only cue that tells one instance of a class from another.
    dull_but_certain = make_detection("cat", 0.95, 4000)
    right_color = make_detection("cat", 0.45, 1000)

    result = select_target(
        [dull_but_certain, right_color],
        "cat",
        requested_color="orange",
        color_scores=[0.0, 0.9],
    )

    assert result is right_color


def test_zero_area_detections_do_not_crash_selection():
    assert select_target([make_detection("cat", 0.8, 0)], "cat") is not None
