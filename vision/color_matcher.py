"""
Color / visual attribute matching module.

Scores how well a detected object's crop matches a requested colour, e.g.
telling an "orange cat" apart from a tabby detected in the same frame.

Every pixel is *classified* into whichever known colour explains it best,
rather than tested against a hand-written range per colour. Ranges were the
original approach and they failed in three ways at once:

- A pixel had to clear a per-colour saturation floor (100 for red/orange/
  yellow) while the denominator counted every pixel above a much lower floor
  of 50. Real objects indoors sit between the two -- an orange bottle at
  S=90 counted as a pixel that *could* match, then matched nothing, so a
  correct target scored a flat 0.00. Every realistic colour did.
- The ranges left gaps. A hue landing between two ranges matched nothing.
- The ranges overlapped. H=172 was both "red" and "pink" at 1.00, so the
  score could not rank one against the other.

Classifying each pixel fixes all three: there is no second threshold to
fall between, the hue circle is covered with no gaps, and the colours are
mutually exclusive by construction.

Pixels are weighted by how central they are in the crop. A bounding box
around a bottle is mostly background at the corners, and weighting the
middle higher keeps that background from drowning out the object. A crop
that is entirely one colour still scores 1.0, since the weights cancel.
"""

import cv2
import numpy as np

# Below this value a pixel is too dark for its hue to mean anything.
BLACK_MAX_VALUE = 55
# Below this saturation a pixel reads as black/white/gray, not as a hue.
ACHROMATIC_MAX_SATURATION = 42
# An achromatic pixel at or above this value is white rather than gray.
WHITE_MIN_VALUE = 165
# Brown has no hue band of its own: it is dark, moderately saturated orange,
# so it is split off from orange by value instead. The saturation ceiling
# keeps a vivid orange that happens to be in shadow from turning brown, and
# the red band is left alone so a shadowed red cup stays red.
BROWN_MAX_VALUE = 145
BROWN_MAX_SATURATION = 200
BROWN_HUES = ("orange",)

ACHROMATIC_COLORS = ("black", "white", "gray")

# Hue centres on OpenCV's 0-179 scale. Every chromatic pixel goes to the
# nearest of these by distance around the hue circle, so the circle is
# covered exactly once -- no gaps, no overlaps.
COLOR_HUES = {
    "red": 0,
    "orange": 15,
    "yellow": 28,
    "green": 62,
    "blue": 110,
    "purple": 140,
    "pink": 167,
}

SUPPORTED_COLORS = frozenset(COLOR_HUES) | set(ACHROMATIC_COLORS) | {"brown"}

# Kept for callers that validate a requested colour against this module's
# vocabulary, and as readable documentation of where each hue band lands.
# Scoring no longer reads it -- see the module docstring for why.
HSV_RANGES = {name: () for name in sorted(SUPPORTED_COLORS)}

# How much less an edge pixel counts than a central one.
MIN_CENTER_WEIGHT = 0.1
CENTER_WEIGHT_INTERCEPT = 1.3

_HUE_NAMES = tuple(COLOR_HUES)
_HUE_CENTERS = np.array([COLOR_HUES[n] for n in _HUE_NAMES], dtype=np.int16)
_HUE_WRAP = 180


def _center_weights(height: int, width: int) -> np.ndarray:
    """Per-pixel weights that fade from the middle of a crop to its corners."""
    # Normalised so the box edge sits at radius 1.0 on each axis.
    ys = (np.arange(height, dtype=np.float32) - (height - 1) / 2) / max(height / 2, 1)
    xs = (np.arange(width, dtype=np.float32) - (width - 1) / 2) / max(width / 2, 1)
    radius = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
    return np.clip(CENTER_WEIGHT_INTERCEPT - radius, MIN_CENTER_WEIGHT, 1.0)


def classify_pixels(hsv: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Label every pixel of an HSV image with the colour that best explains it.

    Returns an index array and the colour name for each index, so callers can
    build any mask they need without re-deriving the thresholds.
    """
    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    names = list(_HUE_NAMES) + ["brown", *ACHROMATIC_COLORS]
    brown_index = len(_HUE_NAMES)
    black_index, white_index, gray_index = brown_index + 1, brown_index + 2, brown_index + 3

    # Nearest hue centre, measured the short way around the 0-179 circle so
    # that red keeps both of its ends without needing two separate ranges.
    gap = np.abs(hue[:, :, None] - _HUE_CENTERS[None, None, :])
    circular_gap = np.minimum(gap, _HUE_WRAP - gap)
    labels = np.argmin(circular_gap, axis=2).astype(np.int8)

    # Dark, muted orange is what people call brown.
    is_brown = (
        np.isin(labels, [_HUE_NAMES.index(n) for n in BROWN_HUES])
        & (value < BROWN_MAX_VALUE)
        & (saturation >= ACHROMATIC_MAX_SATURATION)
        & (saturation <= BROWN_MAX_SATURATION)
    )
    labels[is_brown] = brown_index

    # Hue is noise once a pixel is too washed out or too dark to carry it.
    is_achromatic = saturation < ACHROMATIC_MAX_SATURATION
    labels[is_achromatic] = np.where(
        value[is_achromatic] >= WHITE_MIN_VALUE, white_index, gray_index
    )
    labels[value < BLACK_MAX_VALUE] = black_index

    return labels, names


def color_match_score(crop: np.ndarray, requested_color: str) -> float:
    """
    Return how much of `crop` reads as `requested_color`, from 0.0 to 1.0.

    The result is a centre-weighted fraction of the crop, so a bounding box
    that is half background still scores well when the object itself is the
    right colour.
    """
    if crop is None or crop.size == 0:
        return 0.0

    requested_color = requested_color.strip().lower()
    if requested_color not in SUPPORTED_COLORS:
        raise ValueError(f"Unsupported color: {requested_color}")

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    labels, names = classify_pixels(hsv)
    weights = _center_weights(*hsv.shape[:2])

    # An unlit pixel says nothing about an object's hue, so it neither counts
    # for a chromatic colour nor against it. Left in for black, where being
    # unlit is the whole point.
    if requested_color != "black":
        weights = np.where(hsv[:, :, 2] < BLACK_MAX_VALUE, 0.0, weights)

    total = float(weights.sum())
    if total <= 0.0:
        return 0.0

    matched = float(weights[labels == names.index(requested_color)].sum())
    return matched / total
