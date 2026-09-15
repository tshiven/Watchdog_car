"""
Distance estimation module.

Estimates target distance using the pinhole camera model and a calibrated
camera focal length.
"""


def estimate_distance(
    pixel_width: float,
    real_width_cm: float,
    focal_length_px: float,
) -> float | None:
    """Estimate object distance in centimeters using the pinhole camera model."""
    if pixel_width <= 0:
        return None

    if real_width_cm <= 0:
        return None

    if focal_length_px <= 0:
        return None

    return (real_width_cm * focal_length_px) / pixel_width