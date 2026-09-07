"""
Classical (non-ML) HSV colour blob tracker.

This is the simplest thing that can track a coloured object: threshold the
frame in HSV, clean up the mask, and take the largest remaining blob. It
needs no model and no GPU, so it stays useful as a debug mode after YOLO
lands -- if this works but YOLO does not, the problem is the model, not the
camera or the colour ranges.

HSV ranges here are starting points; use calibration/hsv_tuner.py to find
values that suit your lighting.
"""

import argparse
from dataclasses import dataclass

import cv2
import numpy as np

MIN_CONTOUR_AREA = 500
MORPH_KERNEL_SIZE = 5

# OpenCV HSV ranges: H is 0-179, S and V are 0-255.
# Red straddles H=0, so it needs two ranges rather than one.
HSV_COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "orange": [((10, 120, 70), (25, 255, 255))],
    "yellow": [((25, 100, 100), (35, 255, 255))],
    "green": [((35, 80, 60), (85, 255, 255))],
    "blue": [((90, 100, 60), (130, 255, 255))],
    "purple": [((130, 60, 50), (160, 255, 255))],
    "pink": [((160, 60, 120), (175, 255, 255))],
}


@dataclass
class Blob:
    """A tracked colour blob, in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    center_x: int
    center_y: int
    area: int


def build_mask(frame: np.ndarray, color: str) -> np.ndarray:
    """Return a cleaned binary mask of the pixels matching `color`."""
    if color not in HSV_COLOR_RANGES:
        raise ValueError(
            f"Unknown colour '{color}'. Known: {', '.join(sorted(HSV_COLOR_RANGES))}"
        )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_COLOR_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_largest_blob(mask: np.ndarray, min_area: int = MIN_CONTOUR_AREA) -> Blob | None:
    """Return the largest blob in `mask`, or None if none is big enough."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(largest))
    if area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return Blob(
        x=x,
        y=y,
        width=w,
        height=h,
        center_x=x + w // 2,
        center_y=y + h // 2,
        area=area,
    )


def track_color(
    frame: np.ndarray, color: str, min_area: int = MIN_CONTOUR_AREA
) -> tuple[Blob | None, np.ndarray]:
    """Find the largest blob of `color` in `frame`. Returns (blob, mask)."""
    mask = build_mask(frame, color)
    return find_largest_blob(mask, min_area), mask


def draw_blob(frame: np.ndarray, blob: Blob, label: str = "") -> None:
    """Draw a blob's box and centre onto `frame`, in place."""
    cv2.rectangle(
        frame, (blob.x, blob.y), (blob.x + blob.width, blob.y + blob.height), (0, 255, 0), 2
    )
    cv2.circle(frame, (blob.center_x, blob.center_y), 4, (0, 0, 255), -1)
    text = label or f"area={blob.area}"
    cv2.putText(
        frame, text, (blob.x, max(blob.y - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )


def _run_demo() -> None:
    """Live classical colour tracking, for eyeballing HSV ranges."""
    from vision.camera import Camera, CameraError

    parser = argparse.ArgumentParser(description="Classical HSV colour tracker demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--color", default="orange", choices=sorted(HSV_COLOR_RANGES))
    parser.add_argument("--min-area", type=int, default=MIN_CONTOUR_AREA)
    args = parser.parse_args()

    try:
        cam = Camera(index=args.camera)
    except CameraError as exc:
        print(exc)
        return

    print(f"Tracking '{args.color}'. Press 'q' to quit.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Failed to read frame from camera.")
                break

            blob, mask = track_color(frame, args.color, args.min_area)
            if blob is not None:
                draw_blob(frame, blob, f"{args.color} area={blob.area}")

            cv2.imshow("Watchdog - Classical Tracker", frame)
            cv2.imshow("Watchdog - Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Allow `python vision/classical_color_tracker.py` as well as `python -m ...`
    # by putting the project root on the import path.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _run_demo()
