"""
Color / visual attribute matching module.

Responsible for evaluating how well a detected object's appearance matches
a requested visual attribute, such as a color (e.g. "orange cat" vs a
non-orange cat detected in the same frame).

Will eventually handle:
- Extracting a region of interest from a bounding box
- Converting to HSV and comparing against target color ranges
- Producing a match score usable for ranking candidates

Not yet implemented.
"""

"""
Color / visual attribute matching module.

Provides HSV-based color matching for detected object image crops.
"""

import cv2
import numpy as np


MIN_SATURATION = 50
MIN_VALUE = 50

# OpenCV hue values range from 0 to 179.
# Red requires two ranges because its hue wraps around 0/179.
HSV_RANGES = {
    "red": [
        (np.array([0, 100, 50]), np.array([10, 255, 255])),
        (np.array([170, 100, 50]), np.array([179, 255, 255])),
    ],
    "orange": [
        (np.array([11, 100, 50]), np.array([25, 255, 255])),
    ],
    "yellow": [
        (np.array([26, 100, 50]), np.array([35, 255, 255])),
    ],
    "green": [
        (np.array([36, 70, 50]), np.array([85, 255, 255])),
    ],
    "blue": [
        (np.array([86, 70, 50]), np.array([130, 255, 255])),
    ],
    "purple": [
        (np.array([131, 70, 50]), np.array([160, 255, 255])),
    ],
    "pink": [
        (np.array([161, 50, 50]), np.array([179, 255, 255])),
    ],
    "black": [
        (np.array([0, 0, 0]), np.array([179, 255, 49])),
    ],
    "white": [
        (np.array([0, 0, 200]), np.array([179, 49, 255])),
    ],
    "gray": [
        (np.array([0, 0, 50]), np.array([179, 49, 199])),
    ],
    "brown": [
        (np.array([5, 60, 20]), np.array([25, 255, 180])),
    ],
}


def color_match_score(crop: np.ndarray, requested_color: str) -> float:
    """Return the fraction of valid crop pixels matching the requested color."""
    if crop is None or crop.size == 0:
        return 0.0

    requested_color = requested_color.strip().lower()

    if requested_color not in HSV_RANGES:
        raise ValueError(f"Unsupported color: {requested_color}")

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    valid_mask = (saturation >= MIN_SATURATION) & (value >= MIN_VALUE)

    valid_pixel_count = int(np.count_nonzero(valid_mask))

    if valid_pixel_count == 0:
        return 0.0

    color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in HSV_RANGES[requested_color]:
        color_mask |= cv2.inRange(hsv, lower, upper)

    matching_mask = (color_mask > 0) & valid_mask

    matching_pixel_count = int(np.count_nonzero(matching_mask))

    return matching_pixel_count / valid_pixel_count