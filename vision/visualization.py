"""
Debug overlay drawing.

Draws the pipeline's state onto a frame so it can be checked by eye:
candidate detections, the locked target, the frame center, and the error
vector between them. Every function draws in place and returns the frame.

Takes plain detection dicts and bounding-box tuples, so it stays decoupled
from whatever produced them.
"""

import cv2
import numpy as np

BBox = tuple[int, int, int, int]
Detection = dict

CANDIDATE_COLOR = (0, 200, 200)
TARGET_COLOR = (0, 255, 0)
CENTER_COLOR = (255, 255, 255)
ERROR_COLOR = (0, 0, 255)
HUD_COLOR = (255, 255, 255)

BOX_THICKNESS = 2
TARGET_BOX_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LABEL_MARGIN_PX = 8
CROSSHAIR_ARM_PX = 12
CENTER_DOT_RADIUS_PX = 4
HUD_LINE_HEIGHT_PX = 24
HUD_ORIGIN = (12, 28)


def _draw_label(frame: np.ndarray, text: str, bbox: BBox, color: tuple[int, int, int]) -> None:
    x1, y1, _, _ = bbox
    baseline = max(y1 - LABEL_MARGIN_PX, HUD_ORIGIN[1] // 2)
    cv2.putText(frame, text, (x1, baseline), FONT, FONT_SCALE, color, FONT_THICKNESS)


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Outline every candidate detection with its class name and confidence."""
    for detection in detections:
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), CANDIDATE_COLOR, BOX_THICKNESS)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        _draw_label(frame, label, bbox, CANDIDATE_COLOR)
    return frame


def draw_target(frame: np.ndarray, bbox: BBox, label: str = "TARGET") -> np.ndarray:
    """Highlight the locked target more heavily than the other candidates."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), TARGET_COLOR, TARGET_BOX_THICKNESS)
    _draw_label(frame, label, bbox, TARGET_COLOR)
    cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), CENTER_DOT_RADIUS_PX, TARGET_COLOR, -1)
    return frame


def draw_frame_center(frame: np.ndarray) -> np.ndarray:
    """Draw a crosshair at the middle of the frame -- where the target should end up."""
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    cv2.line(frame, (cx - CROSSHAIR_ARM_PX, cy), (cx + CROSSHAIR_ARM_PX, cy), CENTER_COLOR, 1)
    cv2.line(frame, (cx, cy - CROSSHAIR_ARM_PX), (cx, cy + CROSSHAIR_ARM_PX), CENTER_COLOR, 1)
    return frame


def draw_error_vector(frame: np.ndarray, target_center: tuple[int, int]) -> np.ndarray:
    """Draw the offset the servos have to close, from frame center to target."""
    height, width = frame.shape[:2]
    cv2.arrowedLine(frame, (width // 2, height // 2), target_center, ERROR_COLOR, BOX_THICKNESS)
    return frame


def draw_hud(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Print status lines in the top-left corner."""
    x, y = HUD_ORIGIN
    for offset, line in enumerate(lines):
        position = (x, y + offset * HUD_LINE_HEIGHT_PX)
        cv2.putText(frame, line, position, FONT, FONT_SCALE, HUD_COLOR, FONT_THICKNESS)
    return frame
