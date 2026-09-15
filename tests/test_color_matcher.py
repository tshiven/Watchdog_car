"""Tests for vision/color_matcher.py."""

import cv2
import numpy as np

from vision.color_matcher import SUPPORTED_COLORS, color_match_score


def make_solid_bgr_image(bgr_color: tuple[int, int, int]) -> np.ndarray:
    """Create a 50x50 synthetic image of one BGR color."""
    return np.full((50, 50, 3), bgr_color, dtype=np.uint8)


def test_red_matches_red_but_not_blue():
    red_image = make_solid_bgr_image((0, 0, 255))

    red_score = color_match_score(red_image, "red")
    blue_score = color_match_score(red_image, "blue")

    assert red_score > 0.9
    assert red_score > blue_score


def test_blue_matches_blue_but_not_red():
    blue_image = make_solid_bgr_image((255, 0, 0))

    blue_score = color_match_score(blue_image, "blue")
    red_score = color_match_score(blue_image, "red")

    assert blue_score > 0.9
    assert blue_score > red_score


def test_green_matches_green():
    green_image = make_solid_bgr_image((0, 255, 0))

    green_score = color_match_score(green_image, "green")

    assert green_score > 0.9


def test_orange_matches_orange():
    orange_image = make_solid_bgr_image((0, 165, 255))

    orange_score = color_match_score(orange_image, "orange")
    blue_score = color_match_score(orange_image, "blue")

    assert orange_score > 0.9
    assert orange_score > blue_score


def test_yellow_matches_yellow():
    yellow_image = make_solid_bgr_image((0, 255, 255))

    yellow_score = color_match_score(yellow_image, "yellow")

    assert yellow_score > 0.9


def test_dark_pixels_are_ignored():
    black_image = make_solid_bgr_image((0, 0, 0))

    red_score = color_match_score(black_image, "red")
    blue_score = color_match_score(black_image, "blue")

    assert red_score == 0.0
    assert blue_score == 0.0


def test_empty_crop_returns_zero():
    empty_crop = np.empty((0, 0, 3), dtype=np.uint8)

    assert color_match_score(empty_crop, "red") == 0.0


def test_unsupported_color_raises_error():
    image = make_solid_bgr_image((0, 0, 255))

    try:
        color_match_score(image, "turquoise")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unsupported color")

def hsv_patch(hue: int, saturation: int, value: int, size: int = 50) -> np.ndarray:
    """A solid patch given directly in HSV, for colours as a camera sees them."""
    hsv = np.full((size, size, 3), (hue, saturation, value), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def best_match(image: np.ndarray) -> tuple[str, float]:
    """The colour that scores highest for `image`, and its score."""
    scores = {name: color_match_score(image, name) for name in sorted(SUPPORTED_COLORS)}
    winner = max(scores, key=scores.get)
    return winner, scores[winner]


# Real objects are nowhere near fully saturated. Every one of these scored a
# flat 0.00 under the old range-based matcher, because a pixel had to clear a
# saturation floor of 100 to match while the denominator counted it from 50 --
# so an obviously orange bottle matched nothing at all.
REALISTIC_COLORS = [
    ("orange bottle indoors", (15, 90, 180), "orange"),
    ("red cup in shadow", (3, 95, 120), "red"),
    ("yellow phone case", (30, 85, 200), "yellow"),
    ("green bottle", (65, 140, 150), "green"),
    ("blue bottle, dim", (110, 65, 150), "blue"),
    ("purple case", (145, 65, 160), "purple"),
    ("pink case", (168, 120, 220), "pink"),
    ("white bottle", (0, 15, 175), "white"),
    ("gray laptop", (0, 12, 120), "gray"),
    ("black phone", (0, 10, 30), "black"),
    ("cardboard box", (15, 120, 110), "brown"),
]


def test_realistic_object_colors_are_matched():
    for description, hsv, expected in REALISTIC_COLORS:
        score = color_match_score(hsv_patch(*hsv), expected)

        assert score > 0.9, f"{description} scored {score:.2f} for {expected}"


def test_realistic_object_colors_are_not_confused_with_each_other():
    for description, hsv, expected in REALISTIC_COLORS:
        winner, _ = best_match(hsv_patch(*hsv))

        assert winner == expected, f"{description} was called {winner}"


def test_every_hue_matches_exactly_one_color():
    # The old ranges left gaps between the bands and overlapped at the ends,
    # so some hues matched nothing and others matched two colours at 1.00.
    for hue in range(180):
        image = hsv_patch(hue, 200, 200)
        confident = [
            name for name in sorted(SUPPORTED_COLORS) if color_match_score(image, name) > 0.5
        ]

        assert len(confident) == 1, f"hue {hue} matched {confident}"


def test_shadowed_red_is_red_not_brown():
    assert best_match(hsv_patch(3, 95, 120))[0] == "red"


def test_dark_muted_orange_is_brown():
    assert best_match(hsv_patch(15, 120, 110))[0] == "brown"


def test_vivid_orange_stays_orange_in_shadow():
    assert best_match(hsv_patch(15, 230, 130))[0] == "orange"


def test_background_in_the_bounding_box_does_not_hide_the_object():
    # A bottle occupying the middle of its box, on a mid-gray background --
    # the normal case, since a box is always larger than the thing in it.
    crop = np.full((100, 100, 3), (110, 110, 110), dtype=np.uint8)
    crop[20:80, 32:68] = hsv_patch(15, 180, 200, size=1)[0, 0]

    score = color_match_score(crop, "orange")

    assert score > 0.3, f"object filling 36% of the box scored only {score:.2f}"
    assert color_match_score(crop, "blue") == 0.0


def test_center_pixels_count_for_more_than_edge_pixels():
    centered = np.full((60, 60, 3), (110, 110, 110), dtype=np.uint8)
    centered[20:40, 20:40] = hsv_patch(110, 200, 200, size=1)[0, 0]

    edged = np.full((60, 60, 3), (110, 110, 110), dtype=np.uint8)
    edged[0:20, 0:20] = hsv_patch(110, 200, 200, size=1)[0, 0]

    assert color_match_score(centered, "blue") > color_match_score(edged, "blue")


def test_uniform_crop_scores_one_despite_center_weighting():
    assert color_match_score(hsv_patch(110, 200, 200), "blue") == 1.0
