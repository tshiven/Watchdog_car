"""
Camera capture module.

Wraps cv2.VideoCapture with a small, testable API: open a camera by index,
request a resolution, read frames, and release cleanly. Also provides a
utility to probe which camera indices are available on the current machine.
"""

import argparse

import cv2

DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
MAX_PROBE_INDEX = 3


class CameraError(RuntimeError):
    """Raised when a camera cannot be opened."""


class Camera:
    """A single video capture device."""

    def __init__(
        self,
        index: int = DEFAULT_CAMERA_INDEX,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        self.index = index
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraError(f"Could not open camera index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @property
    def width(self) -> int:
        """Actual frame width the camera is delivering."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Actual frame height the camera is delivering."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Return the next frame, or None if it could not be read."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def probe_camera_indices(max_index: int = MAX_PROBE_INDEX) -> list[int]:
    """Return the indices in [0, max_index] that successfully open."""
    working = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            working.append(i)
        cap.release()
    return working


def _run_demo() -> None:
    """Show a live feed from one camera, or probe available cameras."""
    parser = argparse.ArgumentParser(description="Watchdog camera module demo")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument(
        "--probe", action="store_true", help="Probe camera indices 0-3 and exit"
    )
    args = parser.parse_args()

    if args.probe:
        working = probe_camera_indices()
        print(f"Working camera indices: {working}")
        return

    try:
        cam = Camera(index=args.camera)
    except CameraError as exc:
        print(exc)
        return

    print(f"Opened camera {args.camera} at {cam.width}x{cam.height}. Press 'q' to quit.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Failed to read frame from camera.")
                break
            cv2.imshow("Watchdog - Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_demo()
