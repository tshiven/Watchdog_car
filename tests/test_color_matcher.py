"""
Tests for vision/color_matcher.py.

Placeholder for future tests covering color/attribute match scoring.

Not yet implemented.
"""

"""Tests for vision/color_matcher.py."""

import cv2
import numpy as np

from vision.color_matcher import color_match_score


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