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

import numpy as np
from ultralytics import YOLO

DEFAULT_MODEL = "yolov8n.pt"
# 320/0.50 measured 0/30 detections on a live feed with a person standing in
# frame -- too coarse and too strict for real use. 640/0.25 measured reliable
# person detection (0.94 confidence) on the same feed.
DEFAULT_IMGSZ = 640
DEFAULT_CONFIDENCE = 0.25

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
    ) -> None:
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.confidence = confidence
        self.device = _select_device(prefer_gpu)
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
        return class_name.lower() in {n.lower() for n in self.model.names.values()}

    def detect(self, frame: np.ndarray, target_class: str | None = None) -> list[Detection]:
        """Detect objects in `frame`, optionally keeping only one class."""
        try:
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            if self.device == "cpu":
                print(f"Inference failed: {exc}")
                return []
            print(f"Inference failed on '{self.device}' ({exc}); switching to CPU.")
            self.device = "cpu"
            return self.detect(frame, target_class)

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
