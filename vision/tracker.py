"""
Target tracking module.

Follows the previously selected target across frames by matching each new
frame's detections against the last known bounding box, and reports the
target's pixel offset from the center of the frame -- the signal that will
eventually drive pan/tilt servo control.

Matching is by center distance, gated at a multiple of the target's own
size, rather than by box overlap: a target that is close to the camera can
cross many more pixels between frames than a distant one, and an overlap
test drops a fast mover whose boxes still clearly belong to one object.

Works purely on the shared detection-dict shape, so it can be tested with
hand-built detection lists -- no camera or YOLO model required.
"""

from dataclasses import dataclass

Detection = dict
BBox = tuple[int, int, int, int]

# A target may move up to this multiple of its own mean side length per frame.
DEFAULT_MAX_MOVE_RATIO = 1.0
DEFAULT_MAX_MISSED_FRAMES = 5


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


@dataclass
class TrackResult:
    """Outcome of one Tracker.update() call."""

    bbox: BBox | None
    center: tuple[int, int] | None
    dx: int | None
    dy: int | None
    lost: bool


class Tracker:
    """Follows one target across frames by IoU matching, no re-inference required."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        max_move_ratio: float = DEFAULT_MAX_MOVE_RATIO,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_move_ratio = max_move_ratio
        self.max_missed_frames = max_missed_frames
        self._bbox: BBox | None = None
        self._missed_frames = 0

    @property
    def is_tracking(self) -> bool:
        return self._bbox is not None

    def start(self, detection: Detection) -> None:
        """Begin tracking the given detection."""
        self._bbox = detection["bbox"]
        self._missed_frames = 0

    def stop(self) -> None:
        """Stop tracking; is_tracking becomes False."""
        self._bbox = None
        self._missed_frames = 0

    def update(self, candidates: list[Detection]) -> TrackResult:
        """
        Match `candidates` (this frame's detections, already filtered to the
        target's class) against the last known box and report the target's
        new position, or that it has been lost.
        """
        if self._bbox is None:
            return TrackResult(bbox=None, center=None, dx=None, dy=None, lost=True)

        best = min(candidates, key=lambda d: _center_distance(self._bbox, d["bbox"]), default=None)
        moved = _center_distance(self._bbox, best["bbox"]) if best is not None else 0.0
        max_move = self.max_move_ratio * _mean_side(self._bbox)

        if best is not None and moved <= max_move:
            self._bbox = best["bbox"]
            self._missed_frames = 0
            return self._result_at(best["center"])

        self._missed_frames += 1
        if self._missed_frames > self.max_missed_frames:
            self.stop()
            return TrackResult(bbox=None, center=None, dx=None, dy=None, lost=True)

        # Missed this frame but still within tolerance -- hold the last known position.
        x1, y1, x2, y2 = self._bbox
        return self._result_at(((x1 + x2) // 2, (y1 + y2) // 2))

    def _result_at(self, center: tuple[int, int]) -> TrackResult:
        dx, dy = compute_center_error(center, self.frame_width, self.frame_height)
        return TrackResult(bbox=self._bbox, center=center, dx=dx, dy=dy, lost=False)


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
            draw_detections(frame, detections)
            draw_frame_center(frame)

            status = "SEARCHING"
            if not tracker.is_tracking:
                if detections:
                    tracker.start(max(detections, key=lambda d: d["confidence"]))
            else:
                result = tracker.update(detections)
                if result.lost:
                    status = "LOST"
                elif result.bbox is not None:
                    draw_target(frame, result.bbox, args.target.upper())
                    draw_error_vector(frame, result.center)
                    status = f"LOCKED  dx={result.dx}  dy={result.dy}"

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
