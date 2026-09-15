"""
Target tracking module.

Follows the previously selected target across frames by matching each new
frame's detections against where the target is *predicted* to be, and
reports its pixel offset from the centre of the frame -- the signal that
will eventually drive pan/tilt servo control.

Three things decide whether a candidate is the target: how close it is to
the prediction, whether it is still roughly the same size, and (when the
caller passes the frame in) whether it still looks the same. Appearance is
what keeps a lock on the right bottle when a second one enters the frame,
which position alone cannot do.

The search gate used to be a plain multiple of the target's own size, and
that is why small objects tracked so much worse than people. A person
1.5 m away is ~500 px across, so the gate allowed 500 px of movement per
frame and nothing could outrun it. A bottle is ~110 px across, so the same
rule allowed only 110 px -- less than a hand-carried bottle covers between
frames at 12 fps. Identical motion was followed for a person and dropped
for a bottle. The gate now has a floor tied to the frame size rather than
the object, and it opens along the target's own velocity while the target
is missing, which is when the prediction is least certain.

Matching works on the shared detection-dict shape alone, so it can be
tested with hand-built detection lists -- no camera or YOLO model required.
The frame argument is optional everywhere for that reason.
"""

from dataclasses import dataclass

Detection = dict
BBox = tuple[int, int, int, int]

# A target may move this multiple of its own mean side length per frame...
DEFAULT_MAX_MOVE_RATIO = 2.5
# ...but never less than this fraction of the frame diagonal, so that a small
# object is not held to a tighter gate than a large one crossing the same
# number of pixels.
DEFAULT_MIN_MOVE_FRACTION = 0.10
DEFAULT_MAX_MISSED_FRAMES = 6

# Rejected outright: no real object changes size by more than this in a frame.
MAX_SIZE_RATIO = 3.0

# Relative weight of each cue when ranking candidates inside the gate.
DISTANCE_WEIGHT = 1.0
SIZE_WEIGHT = 0.5
APPEARANCE_WEIGHT = 1.5
# Weight of the caller's own per-candidate prior, such as a colour score.
# Highest of the three: a prior is what the user actually asked for, and
# unlike the appearance model it never drifts, so it is the one cue that can
# still be trusted after the target and a lookalike have crossed paths.
PRIOR_WEIGHT = 2.0

# How fast the appearance model follows lighting and pose changes.
APPEARANCE_BLEND = 0.15
# Below this similarity a candidate is a different object, not a changed one.
# The value is a balance measured over simulated runs rather than a guess:
# raising it drops targets that are partly hidden (a target 70% behind a hand
# still scores about 0.26), and lowering it lets a lookalike take the lock
# after the two have crossed. 0.20 held a correct lock on the largest share
# of frames across fast, slow, occluded and high-dropout runs alike.
MIN_APPEARANCE_SIMILARITY = 0.20

# How much of the previous velocity estimate survives each new measurement.
VELOCITY_SMOOTHING = 0.5
# Coasting bleeds off speed each frame, so a target that is never re-found
# drifts to a stop near where it was last seen instead of flying off-screen.
COASTING_VELOCITY_DECAY = 0.8


def compute_center_error(
    center: tuple[int, int], frame_width: int, frame_height: int
) -> tuple[int, int]:
    """Pixel offset of `center` from the middle of a frame_width x frame_height frame."""
    cx, cy = center
    dx = cx - frame_width // 2
    dy = cy - frame_height // 2
    return dx, dy


def _box_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _center_distance(a: BBox, b: BBox) -> float:
    """Pixel distance between the centers of two boxes."""
    ax, ay = _box_center(a)
    bx, by = _box_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _mean_side(bbox: BBox) -> float:
    """Mean of a box's width and height, used to scale the movement gate."""
    x1, y1, x2, y2 = bbox
    return ((x2 - x1) + (y2 - y1)) / 2


def _shift(bbox: BBox, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy))


def _clamp_speed(velocity: tuple[float, float], limit: float) -> tuple[float, float]:
    """Scale `velocity` down if it exceeds `limit` pixels per frame."""
    vx, vy = velocity
    speed = (vx * vx + vy * vy) ** 0.5
    if speed <= limit or speed == 0:
        return velocity
    scale = limit / speed
    return (vx * scale, vy * scale)


def _size_cost(a: BBox, b: BBox) -> float:
    """0.0 when two boxes are the same size, approaching 1.0 as they diverge."""
    sa, sb = _mean_side(a), _mean_side(b)
    if sa <= 0 or sb <= 0:
        return 1.0
    ratio = max(sa, sb) / min(sa, sb)
    return min((ratio - 1.0) / (MAX_SIZE_RATIO - 1.0), 1.0)


@dataclass
class TrackResult:
    """Outcome of one Tracker.update() call."""

    bbox: BBox | None
    center: tuple[int, int] | None
    dx: int | None
    dy: int | None
    lost: bool
    # True when no detection matched and the position is dead reckoning, so a
    # caller can refuse to steer on a figure the detector has not confirmed.
    coasting: bool = False


class Tracker:
    """Follows one target across frames without re-running selection."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        max_move_ratio: float = DEFAULT_MAX_MOVE_RATIO,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        min_move_fraction: float = DEFAULT_MIN_MOVE_FRACTION,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_move_ratio = max_move_ratio
        self.max_missed_frames = max_missed_frames
        self.min_move_px = min_move_fraction * (frame_width**2 + frame_height**2) ** 0.5
        self._bbox: BBox | None = None
        self._velocity: tuple[float, float] = (0.0, 0.0)
        self._appearance = None
        self._missed_frames = 0

    @property
    def is_tracking(self) -> bool:
        return self._bbox is not None

    def start(self, detection: Detection, frame=None) -> None:
        """Begin tracking the given detection, learning its look if a frame is given."""
        self._bbox = detection["bbox"]
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._appearance = _appearance_of(frame, self._bbox)

    def stop(self) -> None:
        """Stop tracking; is_tracking becomes False."""
        self._bbox = None
        self._velocity = (0.0, 0.0)
        self._appearance = None
        self._missed_frames = 0

    def _predicted_bbox(self) -> BBox:
        """Where the target should be now, carried one frame on at its last velocity."""
        # One step, always: _bbox holds the best estimate of last frame's
        # position, and a coasted frame advances it, so the misses are already
        # accounted for. Multiplying by the miss count here as well would
        # compound them and throw the prediction further out with every frame.
        vx, vy = self._velocity
        return _shift(self._bbox, vx, vy)

    def _base_gate(self) -> float:
        """The gate for a target that was tracked cleanly on the previous frame."""
        return max(self.min_move_px, self.max_move_ratio * _mean_side(self._bbox))

    def _gate(self) -> float:
        """How far from the prediction a detection may be and still be the target."""
        base = self._base_gate()
        # Every frame without a fix makes the prediction less certain, in
        # proportion to how fast the target was going. A target that was
        # standing still gains nothing -- it should not start matching things
        # far away just because it vanished.
        speed = (self._velocity[0] ** 2 + self._velocity[1] ** 2) ** 0.5
        return base + speed * self._missed_frames

    def update(
        self, candidates: list[Detection], frame=None, priors: list[float] | None = None
    ) -> TrackResult:
        """
        Match `candidates` (this frame's detections, already filtered to the
        target's class) against the predicted position and report the target's
        new position, or that it has been lost.

        `priors` optionally scores each candidate from 0.0 to 1.0 on some cue
        the caller knows about and the tracker does not -- in practice how
        well it matches a requested colour. A prior only makes a candidate
        more or less preferred; it never removes one, so a target whose score
        dips for a frame is still available to be matched.
        """
        if self._bbox is None:
            return TrackResult(bbox=None, center=None, dx=None, dy=None, lost=True)

        if priors is not None and len(priors) != len(candidates):
            raise ValueError("priors must match the number of candidates")

        predicted = self._predicted_bbox()
        best = self._best_match(candidates, predicted, frame, priors)

        if best is not None:
            self._adopt(best, frame)
            return self._result_at(best["center"])

        self._missed_frames += 1
        if self._missed_frames > self.max_missed_frames:
            self.stop()
            return TrackResult(bbox=None, center=None, dx=None, dy=None, lost=True)

        # Missed this frame but still within tolerance -- carry the target
        # forward on its last known velocity rather than freezing it, so a
        # target that reappears mid-move is still found near the prediction.
        self._bbox = self._predicted_bbox()
        self._velocity = (
            self._velocity[0] * COASTING_VELOCITY_DECAY,
            self._velocity[1] * COASTING_VELOCITY_DECAY,
        )
        cx, cy = _box_center(self._bbox)
        return self._result_at((int(cx), int(cy)), coasting=True)

    def _best_match(self, candidates, predicted: BBox, frame, priors=None):
        """Lowest-cost candidate inside the gate, or None if nothing qualifies."""
        gate = self._gate()
        best, best_cost = None, float("inf")

        for index, candidate in enumerate(candidates):
            bbox = candidate["bbox"]
            distance = _center_distance(predicted, bbox)
            if distance > gate:
                continue

            size_cost = _size_cost(predicted, bbox)
            if size_cost >= 1.0:
                continue

            cost = DISTANCE_WEIGHT * (distance / gate) + SIZE_WEIGHT * size_cost

            if priors is not None:
                cost += PRIOR_WEIGHT * (1.0 - priors[index])

            similarity = _appearance_similarity(self._appearance, frame, bbox)
            if similarity is not None:
                if similarity < MIN_APPEARANCE_SIMILARITY:
                    continue
                cost += APPEARANCE_WEIGHT * (1.0 - similarity)

            if cost < best_cost:
                best, best_cost = candidate, cost

        return best

    def _adopt(self, detection: Detection, frame) -> None:
        """Take a matched detection as the target's new state."""
        previous_center = _box_center(self._bbox)
        new_bbox = detection["bbox"]
        new_center = _box_center(new_bbox)

        # _bbox was carried forward on every missed frame, so this displacement
        # is already one frame's worth of motion however long the gap was.
        measured = (
            new_center[0] - previous_center[0],
            new_center[1] - previous_center[1],
        )
        smoothed = (
            VELOCITY_SMOOTHING * self._velocity[0] + (1 - VELOCITY_SMOOTHING) * measured[0],
            VELOCITY_SMOOTHING * self._velocity[1] + (1 - VELOCITY_SMOOTHING) * measured[1],
        )
        # A target cannot credibly be moving faster than the gate it just came
        # through; capping keeps one odd jump from launching the prediction.
        self._velocity = _clamp_speed(smoothed, self._base_gate())
        self._bbox = new_bbox
        self._missed_frames = 0
        self._blend_appearance(frame, new_bbox)

    def _blend_appearance(self, frame, bbox: BBox) -> None:
        """Drift the appearance model towards how the target looks right now."""
        observed = _appearance_of(frame, bbox)
        if observed is None:
            return
        if self._appearance is None:
            self._appearance = observed
            return
        self._appearance = (
            (1 - APPEARANCE_BLEND) * self._appearance + APPEARANCE_BLEND * observed
        )

    def _result_at(self, center: tuple[int, int], coasting: bool = False) -> TrackResult:
        dx, dy = compute_center_error(center, self.frame_width, self.frame_height)
        return TrackResult(
            bbox=self._bbox, center=center, dx=dx, dy=dy, lost=False, coasting=coasting
        )


# --- Appearance model -------------------------------------------------------
#
# A hue/saturation histogram of the target's middle. It is deliberately coarse
# and slightly blurred: the target has to survive lighting and pose changes
# between frames, so a fine histogram would reject the very object it is
# meant to recognise. cv2 is imported lazily so the matching logic above stays
# usable -- and testable -- without OpenCV present.

APPEARANCE_HUE_BINS = 16
APPEARANCE_SATURATION_BINS = 4
# Only the middle of a box is sampled; the edges are mostly background.
APPEARANCE_CORE_FRACTION = 0.6
APPEARANCE_MIN_PIXELS = 16


def _core_crop(frame, bbox: BBox):
    """The middle of `bbox`, clipped to the frame, or None if too small."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    inset_x = (x2 - x1) * (1 - APPEARANCE_CORE_FRACTION) / 2
    inset_y = (y2 - y1) * (1 - APPEARANCE_CORE_FRACTION) / 2
    x1, x2 = int(max(0, x1 + inset_x)), int(min(width, x2 - inset_x))
    y1, y2 = int(max(0, y1 + inset_y)), int(min(height, y2 - inset_y))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size >= APPEARANCE_MIN_PIXELS else None


def _appearance_of(frame, bbox: BBox):
    """A normalised hue/saturation histogram of the target, or None."""
    if frame is None:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    crop = _core_crop(frame, bbox)
    if crop is None:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Washed-out and unlit pixels carry no reliable hue, so they are left out
    # rather than piling up in one bin and making everything look alike.
    mask = cv2.inRange(hsv, (0, 40, 40), (179, 255, 255))
    histogram = cv2.calcHist(
        [hsv], [0, 1], mask,
        [APPEARANCE_HUE_BINS, APPEARANCE_SATURATION_BINS], [0, 180, 0, 256],
    )
    histogram = cv2.GaussianBlur(histogram, (3, 3), 0)
    total = histogram.sum()
    if total <= 0:
        return None
    return (histogram / total).astype(np.float32)


def _appearance_similarity(model, frame, bbox: BBox) -> float | None:
    """How alike the model and `bbox` look, 1.0 identical, or None if unknown."""
    if model is None or frame is None:
        return None
    observed = _appearance_of(frame, bbox)
    if observed is None:
        return None

    import cv2
    import numpy as np

    distance = cv2.compareHist(
        np.ascontiguousarray(model, dtype=np.float32),
        np.ascontiguousarray(observed, dtype=np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    )
    return 1.0 - float(distance)


def _run_demo() -> None:
    """Live re-detection-based tracking, for verifying tracker.py by eye."""
    import argparse

    import cv2

    from vision.camera import Camera, CameraError
    from vision.detector import Detector
    from vision.visualization import (
        draw_detections,
        draw_error_vector,
        draw_frame_center,
        draw_hud,
        draw_target,
    )

    parser = argparse.ArgumentParser(description="Tracker demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--target", required=True, help="Class name to lock onto")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    args = parser.parse_args()

    try:
        cam = Camera(index=args.camera)
    except CameraError as exc:
        print(exc)
        return

    overrides = {}
    if args.confidence is not None:
        overrides["confidence"] = args.confidence
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    detector = Detector(**overrides)

    tracker = Tracker(frame_width=cam.width, frame_height=cam.height)
    print(f"Model loaded on device: {detector.device}")
    if not detector.supports_class(args.target):
        print(f"Warning: '{args.target}' is not a class this model knows -- it will never be detected.")
    print("Press 'q' to quit.")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Failed to read frame from camera.")
                break

            detections = detector.detect(frame, args.target)
            # Overlays are drawn on a copy so the appearance model samples the
            # camera's pixels, not the boxes drawn over them.
            clean = frame.copy()
            draw_detections(frame, detections)
            draw_frame_center(frame)

            status = "SEARCHING"
            if not tracker.is_tracking:
                if detections:
                    tracker.start(max(detections, key=lambda d: d["confidence"]), clean)
            else:
                result = tracker.update(detections, clean)
                if result.lost:
                    status = "LOST"
                elif result.bbox is not None:
                    draw_target(frame, result.bbox, args.target.upper())
                    draw_error_vector(frame, result.center)
                    state = "COASTING" if result.coasting else "LOCKED"
                    status = f"{state}  dx={result.dx}  dy={result.dy}"

            draw_hud(frame, [status])
            cv2.imshow("Watchdog - Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _run_demo()
