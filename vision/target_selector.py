"""
Target selection module.

Chooses which detection to lock onto when several candidates of the right
class are in frame, by combining detection confidence, how well each one
matches the requested colour, and how large it is.

The confidence floor used to sit at 0.50 while the detector emitted
everything from 0.25 up. Anything in between was drawn on screen but could
never be locked onto, and that band is exactly where yolov8n puts small
objects: a person scores ~0.94 and is picked instantly, while a bottle at
0.42 or a phone at 0.38 were both refused, so the system sat in SEARCHING
while plainly showing the object it had been asked to find. The floor now
matches the detector's own, and confidence is weighed rather than gated --
a weak detection is a worse candidate, not an impossible one.

Colour outranks confidence when a colour was asked for, because it is the
only cue that separates one instance of a class from another; size and
confidence do no better than chance at telling the orange cat from the
gray one.
"""

from typing import Any

# Matches vision.detector.DEFAULT_CONFIDENCE: anything the detector is
# willing to report is a candidate worth ranking.
MIN_CONFIDENCE = 0.10

DETECTION_WEIGHT = 0.35
COLOR_WEIGHT = 0.45
SIZE_WEIGHT = 0.20


def select_target(
    detections: list[dict[str, Any]],
    requested_class: str,
    requested_color: str | None = None,
    color_scores: list[float] | None = None,
) -> dict[str, Any] | None:
    """Return the highest-scoring detection that meets the selection criteria."""
    if color_scores is not None and len(color_scores) != len(detections):
        raise ValueError("color_scores must match the number of detections")

    wanted_class = requested_class.strip().lower()
    candidates = []

    for index, detection in enumerate(detections):
        # Case-insensitive: the class name comes from the model's own table,
        # and a request typed by a user should not have to match its casing.
        if detection["class_name"].strip().lower() != wanted_class:
            continue

        if detection["confidence"] < MIN_CONFIDENCE:
            continue

        color_score = 0.0
        if requested_color is not None and color_scores is not None:
            color_score = color_scores[index]

        candidates.append((detection, color_score))

    if not candidates:
        return None

    largest_area = max(detection["area"] for detection, _ in candidates)

    best_detection = None
    best_score = float("-inf")

    for detection, color_score in candidates:
        # Guard the ratio: a zero-area box would otherwise divide by zero.
        normalized_area = detection["area"] / largest_area if largest_area > 0 else 0.0

        score = (
            DETECTION_WEIGHT * detection["confidence"]
            + SIZE_WEIGHT * normalized_area
        )

        if requested_color is not None:
            score += COLOR_WEIGHT * color_score

        if score > best_score:
            best_score = score
            best_detection = detection

    return best_detection
