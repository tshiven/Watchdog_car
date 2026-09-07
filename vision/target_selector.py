"""
Target selection module.

Responsible for choosing the single best-matching target when multiple
candidate detections are present, combining detection confidence with
attribute match scores (e.g. color) from color_matcher.

Will eventually handle:
- Ranking candidate detections by combined score
- Selecting and locking onto the best match
- Handling the case where no candidate meets a minimum confidence threshold

Not yet implemented.
"""

"""
Target selection module.

Selects the best detection by combining confidence, optional color
matching, and normalized object size.
"""

from typing import Any


MIN_CONFIDENCE = 0.5

DETECTION_WEIGHT = 0.5
COLOR_WEIGHT = 0.3
SIZE_WEIGHT = 0.2


def select_target(
    detections: list[dict[str, Any]],
    requested_class: str,
    requested_color: str | None = None,
    color_scores: list[float] | None = None,
) -> dict[str, Any] | None:
    """Return the highest-scoring detection that meets the selection criteria."""
    if color_scores is not None and len(color_scores) != len(detections):
        raise ValueError("color_scores must match the number of detections")

    candidates = []

    for index, detection in enumerate(detections):
        if detection["class_name"] != requested_class:
            continue

        if detection["confidence"] < MIN_CONFIDENCE:
            continue

        color_score = 0.0
        if requested_color is not None and color_scores is not None:
            color_score = color_scores[index]

        candidates.append((index, detection, color_score))

    if not candidates:
        return None

    largest_area = max(
        detection["area"] for _, detection, _ in candidates
    )

    best_detection = None
    best_score = float("-inf")

    for _, detection, color_score in candidates:
        normalized_area = detection["area"] / largest_area

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
