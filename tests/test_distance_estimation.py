"""Tests for vision/distance_estimation.py."""

from vision.distance_estimation import estimate_distance


def test_estimate_distance():
    distance = estimate_distance(
        pixel_width=65,
        real_width_cm=6.5,
        focal_length_px=800,
    )

    assert distance == 80.0


def test_larger_pixel_width_means_closer_distance():
    closer = estimate_distance(
        pixel_width=100,
        real_width_cm=10,
        focal_length_px=800,
    )

    farther = estimate_distance(
        pixel_width=50,
        real_width_cm=10,
        focal_length_px=800,
    )

    assert closer < farther


def test_zero_pixel_width_returns_none():
    result = estimate_distance(
        pixel_width=0,
        real_width_cm=6.5,
        focal_length_px=800,
    )

    assert result is None


def test_negative_pixel_width_returns_none():
    result = estimate_distance(
        pixel_width=-10,
        real_width_cm=6.5,
        focal_length_px=800,
    )

    assert result is None


def test_invalid_real_width_returns_none():
    result = estimate_distance(
        pixel_width=65,
        real_width_cm=0,
        focal_length_px=800,
    )

    assert result is None


def test_invalid_focal_length_returns_none():
    result = estimate_distance(
        pixel_width=65,
        real_width_cm=6.5,
        focal_length_px=0,
    )

    assert result is None