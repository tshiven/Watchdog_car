"""
Interactive HSV range finder.

Shows the live camera feed next to the binary mask produced by the current
trackbar settings, so colour ranges can be measured under real lighting
instead of guessed. On exit the chosen bounds are printed in a form that can
be pasted straight into HSV_COLOR_RANGES in vision/classical_color_tracker.py.

Run:
    python -m calibration.hsv_tuner --camera 0 --color orange
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running this file directly, not just as `python -m calibration.hsv_tuner`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision.camera import Camera, CameraError
from vision.classical_color_tracker import HSV_COLOR_RANGES

CONTROLS_WINDOW = "Watchdog - HSV Controls"
TRACKBARS = [
    ("H min", 179),
    ("H max", 179),
    ("S min", 255),
    ("S max", 255),
    ("V min", 255),
    ("V max", 255),
]


def _noop(_value: int) -> None:
    """Trackbar callbacks are required but we read values on demand."""


def _create_controls(initial: dict[str, int]) -> None:
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 420, 260)
    for name, maximum in TRACKBARS:
        cv2.createTrackbar(name, CONTROLS_WINDOW, initial.get(name, 0), maximum, _noop)


def _read_controls() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    values = {name: cv2.getTrackbarPos(name, CONTROLS_WINDOW) for name, _ in TRACKBARS}
    lower = (values["H min"], values["S min"], values["V min"])
    upper = (values["H max"], values["S max"], values["V max"])
    return lower, upper


def _starting_values(color: str | None) -> dict[str, int]:
    """Seed the trackbars from a known colour, or with a permissive default."""
    if color and color in HSV_COLOR_RANGES:
        lower, upper = HSV_COLOR_RANGES[color][0]
        return {
            "H min": lower[0], "S min": lower[1], "V min": lower[2],
            "H max": upper[0], "S max": upper[1], "V max": upper[2],
        }
    return {"H min": 0, "S min": 0, "V min": 0, "H max": 179, "S max": 255, "V max": 255}


def _run_tuner() -> None:
    parser = argparse.ArgumentParser(description="Interactive HSV range tuner")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--color", default=None, choices=sorted(HSV_COLOR_RANGES),
        help="Seed the sliders with this colour's current range",
    )
    args = parser.parse_args()

    try:
        cam = Camera(index=args.camera)
    except CameraError as exc:
        print(exc)
        return

    _create_controls(_starting_values(args.color))
    print("Adjust the sliders until only your object is white in the mask.")
    print("Press 'q' to quit and print the final values.")

    lower: tuple[int, int, int] | None = None
    upper: tuple[int, int, int] | None = None
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Failed to read frame from camera.")
                break

            lower, upper = _read_controls()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

            cv2.imshow("Watchdog - Original", frame)
            cv2.imshow("Watchdog - Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    if lower is not None and upper is not None:
        print("\nFinal HSV range:")
        print(f"    lower = {lower}")
        print(f"    upper = {upper}")
        print("\nPaste into HSV_COLOR_RANGES in vision/classical_color_tracker.py as:")
        print(f'    "<colour>": [({lower}, {upper})],')


if __name__ == "__main__":
    _run_tuner()
