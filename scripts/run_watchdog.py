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
import struct
import time
from dataclasses import dataclass

import cv2
import serial

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
SERIAL_PORT = "COM3"
SERIAL_BAUDRATE = 115200


@dataclass
class FrameOutcome:
    """What the pipeline made of one frame."""

    detections: list[dict]
    status: str
    bbox: tuple[int, int, int, int] | None = None
    center: tuple[int, int] | None = None
    dx: int | None = None
    dy: int | None = None
    # True while the target's position is predicted rather than detected.
    coasting: bool = False


def crop_to_bbox(frame, bbox: tuple[int, int, int, int]):
    """The pixels inside bbox, clipped to the frame's bounds."""
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    return frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]


def process_frame(frame, detector, tracker, target_class: str, target_color: str | None) -> FrameOutcome:
    """Run one frame through detect -> select -> track."""
    detections = detector.detect(frame)
    wanted = target_class.strip().lower()
    candidates = [d for d in detections if d["class_name"].strip().lower() == wanted]

    # Scored for every candidate on every frame, but used as a preference
    # rather than a filter -- see the note below.
    priors = None
    if target_color is not None:
        priors = [
            color_match_score(crop_to_bbox(frame, d["bbox"]), target_color)
            for d in candidates
        ]

    if not tracker.is_tracking:
        # Locking on is the one place a hard colour threshold belongs: with no
        # target yet there is nothing else to say which object was meant.
        lockable, color_scores = candidates, None
        if priors is not None:
            matching = [(d, s) for d, s in zip(candidates, priors) if s >= MIN_COLOR_MATCH]
            lockable = [d for d, _ in matching]
            color_scores = [s for _, s in matching]

        target = select_target(lockable, target_class, target_color, color_scores)
        if target is None:
            return FrameOutcome(detections=detections, status=STATUS_SEARCHING)
        tracker.start(target, frame)

    # Every candidate of the right class goes to the tracker, with its colour
    # score attached as a preference. Re-filtering by colour on each frame
    # used to hide the target from the tracker whenever its score dipped -- a
    # highlight, a shadow, a hand across it -- and a few such frames in a row
    # ended the lock on an object that had not gone anywhere. Passing the
    # score instead keeps colour telling the target apart from a lookalike,
    # which is what it was there for, without it ever being able to make the
    # target disappear.
    result = tracker.update(candidates, frame, priors)
    if result.lost:
        return FrameOutcome(detections=detections, status=STATUS_LOST)
    return FrameOutcome(
        detections=detections,
        status=STATUS_LOCKED,
        bbox=result.bbox,
        center=result.center,
        dx=result.dx,
        dy=result.dy,
        coasting=result.coasting,
    )


def render(frame, outcome: FrameOutcome, label: str):
    """Draw the pipeline's state onto the frame."""
    draw_detections(frame, outcome.detections)
    draw_frame_center(frame)
    if outcome.bbox is not None:
        draw_target(frame, outcome.bbox, label.upper())
        draw_error_vector(frame, outcome.center)
        state = "COASTING" if outcome.coasting else outcome.status
        draw_hud(frame, [f"{state}  err_x={outcome.dx}  err_y={outcome.dy}"])
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


def track_until_stopped(cam, detector, ser, target_class, target_color, show_display, verbose=False) -> None:
    """Follow one target until the user stops it or the camera gives out."""
    label = f"{target_color} {target_class}" if target_color else target_class
    print(f"\nTarget: {label}")
    print("Press 'q' in the window to stop." if show_display else "Press Ctrl+C to stop.")

    tracker = Tracker(frame_width=cam.width, frame_height=cam.height)
    last_status = None
    dropped_frames = 0
    serial_rx_buffer = b""
    serial_response_received = False
    serial_warning_printed = False
    serial_wait_started = time.monotonic()
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

            # Printing per frame is not free: at 30 fps a few lines a frame is
            # enough terminal I/O to cut the frame rate, and a slower loop means
            # the target moves further between detections -- the tracker's
            # hardest case. Off unless asked for.
            if verbose and outcome.status == STATUS_LOCKED:
                print(f"err_x={outcome.dx}, err_y={outcome.dy}")

            if ser is not None:
                locked = outcome.status == STATUS_LOCKED
                err_x = max(-32768, min(32767, int(outcome.dx if locked else 0)))
                err_y = max(-32768, min(32767, int(outcome.dy if locked else 0)))
                payload = struct.pack("<hh", err_x, err_y)
                checksum = 4
                for byte in payload:
                    checksum ^= byte
                packet = b"\xAA\x55" + bytes([4]) + payload + bytes([checksum])
                if verbose:
                    print(f"VISION TX: err_x={err_x} err_y={err_y}")
                try:
                    sent = ser.write(packet)
                    if verbose:
                        print(f"SENT {sent} BYTES")
                except Exception as exc:
                    print(f"SERIAL WRITE ERROR: {exc!r}")
                    ser.close()
                    ser = None

                if ser is not None and ser.in_waiting > 0:
                    try:
                        while ser.in_waiting > 0:
                            serial_rx_buffer += ser.readline()
                            if b"\n" not in serial_rx_buffer:
                                break
                            complete_lines = serial_rx_buffer.split(b"\n")
                            serial_rx_buffer = complete_lines.pop()
                            for line in complete_lines:
                                received_line = line.rstrip(b"\r").decode(errors="replace")
                                print(f"STM32 RX: {received_line}")
                                serial_response_received = True
                    except Exception as exc:
                        print(f"SERIAL READ ERROR: {exc!r}")

                if (
                    not serial_response_received
                    and not serial_warning_printed
                    and time.monotonic() - serial_wait_started >= 2.0
                ):
                    print("WARNING: no STM32 response received")
                    serial_warning_printed = True

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
    parser.add_argument("--port", default=SERIAL_PORT, help="Serial port for the STM32")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--no-display", action="store_true", help="Run headless, without a preview window")
    parser.add_argument("--verbose", action="store_true", help="Print the centre error and serial traffic every frame")
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

    ser = None
    if args.port is not None:
        print("SERIAL INIT START")
        try:
            ser = serial.Serial(args.port, SERIAL_BAUDRATE, timeout=0.0)
            print(f"SERIAL OPEN {args.port} {SERIAL_BAUDRATE}")
        except Exception as exc:
            print(f"SERIAL OPEN ERROR: {exc!r}")
            print("Continuing without the serial link.")

    show_display = not args.no_display
    print(
        f"Model on {detector.device}, camera {cam.width}x{cam.height}, "
        f"serial {'-> ' + args.port if ser is not None else 'disabled'}."
    )

    try:
        if args.command is not None:
            target = resolve_target(args.command, detector)
            if target is not None:
                track_until_stopped(cam, detector, ser, *target, show_display, args.verbose)
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
                track_until_stopped(cam, detector, ser, *target, show_display, args.verbose)
    finally:
        print("Shutting down.")
        cam.release()
        if ser is not None:
            ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
