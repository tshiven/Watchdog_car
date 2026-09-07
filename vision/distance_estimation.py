"""
Distance estimation module.

Responsible for estimating the real-world distance to the tracked target,
to eventually support range-aware behavior (e.g. stopping distance, IMU
fusion for stabilization).

Will eventually handle:
- Estimating distance from bounding box size and camera calibration data
- Possibly fusing with IMU or stereo/depth data in the future

Not yet implemented.
"""

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