"""
YOLO object detection.

Loads the model once and turns Ultralytics results into plain dicts, so the
rest of the pipeline never touches an Ultralytics object. Every detection
looks like:

    {
        "class_id": int,
        "class_name": str,
        "confidence": float,
        "bbox": (x1, y1, x2, y2),
        "center": (cx, cy),
        "width": int,
        "height": int,
        "area": int,
    }

Apple Silicon GPU (MPS) is used when it works, with a silent fall back to CPU
otherwise -- an unavailable GPU should never stop the program from running.
"""

import argparse
import time

import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
# 320 rather than 640, for the Raspberry Pi 4. The note this replaces recorded
# "320/0.50 measured 0/30 detections ... 640/0.25 measured reliable person
# detection": two variables moved at once, and the confidence floor is the one
# that explains the result. 0.50 is above where yolov8n puts anything but a
# large, close, unambiguous object -- it is the same floor target_selector.py
# had to drop to 0.25 for a bottle at 0.42 to be lockable at all. 320 was never
# measured at the 0.25 this file now defaults to.
#
# What it buys: yolov8n on a Pi 4 CPU runs roughly 4x faster at 320 than at
# 640, which is the difference between a vision update every ~500 ms and every
# ~150 ms. The firmware's stale-packet failsafe, the pan/tilt rate scaling and
# the follower all key off that interval, so it is not a detail -- see
# FAILSAFE_TIMEOUT_MS in the firmware's main.c.
#
# What it costs: less detail per object, so small or distant targets are found
# less reliably. A person -- the case the car is built around -- is large and
# easy and survives the drop comfortably; a bottle across the room may not.
# `--imgsz 640` restores the old behaviour exactly, and the PERF line the
# runner prints is how to tell whether 320 was needed in the first place.
DEFAULT_IMGSZ = 320
# Dropped from 0.25 to 0.10: on the Pi 4's ~1 FPS loop, waiting for a
# confident detection costs whole seconds of not tracking. A weak detection
# now still locks (see LOCKED in run_watchdog.py), so a lower floor trades
# some false positives for a car that actually follows its target.
DEFAULT_CONFIDENCE = 0.10
# Overlap above which two boxes of the same class are treated as one object.
DEFAULT_IOU = 0.45
DEFAULT_MAX_DETECTIONS = 50

Detection = dict


def _select_device(prefer_gpu: bool) -> str:
    """Return 'mps' if the Apple GPU is usable, else 'cpu'."""
    if not prefer_gpu:
        return "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch missing or MPS check blew up -- CPU is fine
        pass
    return "cpu"


class Detector:
    """Runs YOLO on frames and returns plain-dict detections."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        imgsz: int = DEFAULT_IMGSZ,
        confidence: float = DEFAULT_CONFIDENCE,
        prefer_gpu: bool = True,
        iou: float = DEFAULT_IOU,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
    ) -> None:
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections
        self.device = _select_device(prefer_gpu)
        # How long the last detect() call spent inside the model, in
        # milliseconds. Read by the runner's PERF line to separate "inference
        # is slow" from "everything else is slow", which have different fixes.
        # 0.0 until the first real inference.
        self.last_inference_ms = 0.0
        if self.device != "cpu" and not self._device_works():
            print(f"Device '{self.device}' failed a warm-up run; falling back to CPU.")
            self.device = "cpu"

    def _device_works(self) -> bool:
        """Run one throwaway inference to prove the chosen device works."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        try:
            self.model.predict(
                dummy, imgsz=self.imgsz, device=self.device, verbose=False
            )
            return True
        except Exception as exc:
            print(f"Warm-up on '{self.device}' failed: {exc}")
            return False

    @property
    def class_names(self) -> dict[int, str]:
        """Class id -> name, as the loaded model knows them."""
        return self.model.names

    def supports_class(self, class_name: str) -> bool:
        return self.class_id(class_name) is not None

    def class_id(self, class_name: str) -> int | None:
        """The model's id for `class_name`, or None if it does not know it."""
        wanted = class_name.strip().lower()
        for class_id, name in self.model.names.items():
            if name.strip().lower() == wanted:
                return int(class_id)
        return None

    def detect(self, frame: np.ndarray, target_class: str | None = None) -> list[Detection]:
        """Detect objects in `frame`, optionally keeping only one class."""
        # Narrowing the model to the requested class is not the same as
        # dropping the other classes afterwards: it also keeps a confident
        # overlapping box of another class from suppressing the one we want,
        # which is how a phone held in a hand or a bottle held against a
        # body goes missing.
        wanted_id = self.class_id(target_class) if target_class else None
        started = time.perf_counter()
        try:
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_detections,
                classes=None if wanted_id is None else [wanted_id],
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            if self.device == "cpu":
                print(f"Inference failed: {exc}")
                self.last_inference_ms = 0.0
                return []
            print(f"Inference failed on '{self.device}' ({exc}); switching to CPU.")
            self.device = "cpu"
            # The retry sets the timing itself, so it is not overwritten here.
            return self.detect(frame, target_class)
        self.last_inference_ms = 1000.0 * (time.perf_counter() - started)

        wanted = target_class.lower() if target_class else None
        detections: list[Detection] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            if wanted and class_name.lower() != wanted:
                continue
            detections.append(_to_detection(box, class_id, class_name))
        return detections


def _to_detection(box, class_id: int, class_name: str) -> Detection:
    """Convert one Ultralytics box into the shared detection dict."""
    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
    width, height = x2 - x1, y2 - y1
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": float(box.conf[0]),
        "bbox": (x1, y1, x2, y2),
        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        "width": width,
        "height": height,
        "area": width * height,
    }


def _run_demo() -> None:
    """Live YOLO detection with simple boxes drawn, for verifying Phase 3."""
    import cv2

    from vision.camera import Camera, CameraError

    parser = argparse.ArgumentParser(description="YOLO detector demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--target", default=None, help="Only show this class")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    detector = Detector(
        imgsz=args.imgsz, confidence=args.confidence, prefer_gpu=not args.cpu
    )
    print(f"Model loaded on device: {detector.device}")
    if args.target and not detector.supports_class(args.target):
        print(f"Warning: '{args.target}' is not a class this model knows.")

    try:
        cam = Camera(index=args.camera)
    except CameraError as exc:
        print(exc)
        return

    print("Press 'q' to quit.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                print("Failed to read frame from camera.")
                break

            for det in detector.detect(frame, args.target):
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
                cv2.circle(frame, det["center"], 4, (0, 0, 255), -1)

            cv2.imshow("Watchdog - Detector", frame)
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
