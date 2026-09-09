"""
Main entry point for running the Watchdog system end-to-end.

Asks what to look for, then for every frame: detect candidates, score them
against the requested colour, lock onto the best one, follow it, and report
how far it sits from the centre of the frame.

The centre error goes to the STM32 over serial when --port is given. Without
one the whole pipeline still runs and reports normally, it just sends
nothing -- so the vision half is fully usable before any hardware exists.

    python scripts/run_watchdog.py
    python scripts/run_watchdog.py --command "find the orange cat"
    python scripts/run_watchdog.py --command "find the cup" --no-display
"""

if __name__ == "__main__":  # Allow running this file directly, not just with -m.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import dataclass

import cv2

from communication.serial_protocol import SerialError, SerialLink
from interface.command_parser import parse_command
from vision.camera import Camera, CameraError
from vision.color_matcher import HSV_RANGES, color_match_score
from vision.detector import Detector
from vision.target_selector import select_target
from vision.tracker import Tracker
from vision.visualization import (
    draw_detections,
    draw_error_vector,
    draw_frame_center,
    draw_hud,
    draw_target,
)

WINDOW_TITLE = "Watchdog"
# Cameras drop frames -- measured a None on the very first read while warming
# up, then 299/300 good frames. Only give up once they stop coming entirely.
MAX_CONSECUTIVE_FRAME_FAILURES = 30
# A candidate showing less than this fraction of the requested colour is not
# the target, however confident or large it is. Low enough to survive a bbox
# that includes background and a target that is partly shadowed.
MIN_COLOR_MATCH = 0.15
STATUS_SEARCHING = "SEARCHING"
STATUS_LOCKED = "LOCKED"
STATUS_LOST = "LOST"


@dataclass
class FrameOutcome:
    """What the pipeline made of one frame."""

    detections: list[dict]
    status: str
    bbox: tuple[int, int, int, int] | None = None
    center: tuple[int, int] | None = None
    dx: int | None = None
    dy: int | None = None


def crop_to_bbox(frame, bbox: tuple[int, int, int, int]):
    """The pixels inside bbox, clipped to the frame's bounds."""
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    return frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]


def process_frame(frame, detector, tracker, target_class: str, target_color: str | None) -> FrameOutcome:
    """Run one frame through detect -> select -> track."""
    detections = detector.detect(frame)
    candidates = [d for d in detections if d["class_name"] == target_class]

    color_scores = None
    if target_color is not None:
        # Scored every frame, not just the one that locks on: the tracker matches
        # on position alone, so a wrong-coloured object passing near the target
        # would otherwise capture the lock and keep it for good.
        scored = [
            (d, color_match_score(crop_to_bbox(frame, d["bbox"]), target_color))
            for d in candidates
        ]
        matching = [(d, score) for d, score in scored if score >= MIN_COLOR_MATCH]
        candidates = [d for d, _ in matching]
        color_scores = [score for _, score in matching]

    if not tracker.is_tracking:
        target = select_target(candidates, target_class, target_color, color_scores)
        if target is None:
            return FrameOutcome(detections=detections, status=STATUS_SEARCHING)
        tracker.start(target)

    result = tracker.update(candidates)
    if result.lost:
        return FrameOutcome(detections=detections, status=STATUS_LOST)
    return FrameOutcome(
        detections=detections,
        status=STATUS_LOCKED,
        bbox=result.bbox,
        center=result.center,
        dx=result.dx,
        dy=result.dy,
    )


def render(frame, outcome: FrameOutcome, label: str):
    """Draw the pipeline's state onto the frame."""
    draw_detections(frame, outcome.detections)
    draw_frame_center(frame)
    if outcome.bbox is not None:
        draw_target(frame, outcome.bbox, label.upper())
        draw_error_vector(frame, outcome.center)
        draw_hud(frame, [f"{outcome.status}  err_x={outcome.dx}  err_y={outcome.dy}"])
    else:
        draw_hud(frame, [f"{outcome.status}  {label}"])
    return frame


def _announce(outcome: FrameOutcome, label: str) -> None:
    """Print a one-line note when the pipeline changes state."""
    if outcome.status == STATUS_LOCKED:
        print(f"Locked onto {label}  (dx={outcome.dx}, dy={outcome.dy})")
    elif outcome.status == STATUS_LOST:
        print("Lost the target; searching again.")
    else:
        print(f"Searching for {label}...")


def resolve_target(command: str, detector) -> tuple[str, str | None] | None:
    """Turn a command into (class, colour), or None with a reason printed."""
    parsed = parse_command(command)
    target_class, target_color = parsed["target_class"], parsed["target_color"]

    if target_class is None:
        print("I couldn't identify an object to find. Try 'find the orange cat'.")
        return None
    if not detector.supports_class(target_class):
        print(f"'{target_class}' is not a class this model knows, so it can never be found.")
        return None
    if target_color is not None and target_color not in HSV_RANGES:
        print(f"'{target_color}' is not a colour I can match; ignoring it.")
        target_color = None
    return target_class, target_color


def track_until_stopped(cam, detector, link, target_class, target_color, show_display) -> None:
    """Follow one target until the user stops it or the camera gives out."""
    label = f"{target_color} {target_class}" if target_color else target_class
    print(f"\nTarget: {label}")
    print("Press 'q' in the window to stop." if show_display else "Press Ctrl+C to stop.")

    tracker = Tracker(frame_width=cam.width, frame_height=cam.height)
    last_status = None
    dropped_frames = 0
    try:
        while True:
            frame = cam.read()
            if frame is None:
                dropped_frames += 1
                if dropped_frames > MAX_CONSECUTIVE_FRAME_FAILURES:
                    print("Camera stopped returning frames.")
                    break
                continue
            dropped_frames = 0

            outcome = process_frame(frame, detector, tracker, target_class, target_color)

            if outcome.status == STATUS_LOCKED:
                print(f"err_x={outcome.dx}, err_y={outcome.dy}")

            if link is not None:
                locked = outcome.status == STATUS_LOCKED
                link.send_center_error(
                    outcome.dx if locked else 0, outcome.dy if locked else 0, locked
                )

            if outcome.status != last_status:
                _announce(outcome, label)
                last_status = outcome.status

            if show_display:
                cv2.imshow(WINDOW_TITLE, render(frame, outcome, label))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print()
    finally:
        if show_display:
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Watchdog tracking system")
    parser.add_argument("--command", default=None, help='e.g. "find the orange cat"; prompts if omitted')
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--port", default=None, help="Serial port for the STM32; omit to run without it")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--no-display", action="store_true", help="Run headless, without a preview window")
    args = parser.parse_args()

    overrides = {}
    if args.confidence is not None:
        overrides["confidence"] = args.confidence
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    detector = Detector(**overrides)

    try:
        cam = Camera(index=args.camera)
    except CameraError as exc:
        print(exc)
        return

    link = None
    if args.port is not None:
        try:
            link = SerialLink(args.port)
        except SerialError as exc:
            print(f"{exc}\nContinuing without the serial link.")

    show_display = not args.no_display
    print(
        f"Model on {detector.device}, camera {cam.width}x{cam.height}, "
        f"serial {'-> ' + args.port if link is not None else 'disabled'}."
    )

    try:
        if args.command is not None:
            target = resolve_target(args.command, detector)
            if target is not None:
                track_until_stopped(cam, detector, link, *target, show_display)
            return

        print("\nWATCHDOG -- type what to find, or 'quit' to exit.")
        print('Examples: "find the orange cat", "track the red bottle", "person"')
        while True:
            try:
                command = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not command:
                continue
            if command.lower() in {"quit", "exit", "q"}:
                break

            target = resolve_target(command, detector)
            if target is not None:
                track_until_stopped(cam, detector, link, *target, show_display)
    finally:
        print("Shutting down.")
        cam.release()
        if link is not None:
            link.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
