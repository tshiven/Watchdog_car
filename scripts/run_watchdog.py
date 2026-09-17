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
import time
import traceback
from dataclasses import dataclass

import cv2
import serial

from communication.packet_utils import (
    # Aliased because this file also has string pipeline states below, one of
    # which is literally named STATUS_LOCKED. Imported under their own names
    # the string shadowed the 0x02 bit, and building the status byte then ran
    # `"LOCKED" | 0x00` -- a TypeError on the first locked frame. These two are
    # the wire bits; STATUS_SEARCHING/LOCKED/LOST below are pipeline states.
    STATUS_DETECTED as STATUS_BIT_DETECTED,
    STATUS_LOCKED as STATUS_BIT_LOCKED,
    clamp_size_pct,
    encode_target_name_packet,
    encode_tracking_packet,
    normalize_target_name,
)
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
# The STM32 enumerates as a USB CDC / ST-Link VCP device on the Pi, which is
# /dev/ttyACM0. This used to default to the Windows name "COM3", which can
# never open on Linux -- and a failed open was only a printed warning, so the
# run continued with ser = None and transmitted nothing while vision went on
# reporting detected/locked/status perfectly. Override with --port.
SERIAL_PORT = "/dev/ttyACM0"
SERIAL_BAUDRATE = 115200
# At the Pi 4's ~1 FPS, waiting for a run of consecutive detections before
# moving costs whole seconds of the car standing still, so DETECTED -- which
# is what authorises the drivetrain -- asserts on the first fresh
# (non-coasting) detection. LOCKED is the confirmation on top of that and
# takes this many consecutive fresh detections, i.e. roughly one more frame.
LOCK_FRAMES = 2
# Cadence of the status debug line, in processed frames.
STATUS_DEBUG_EVERY = 12
# Cadence of the timing line, in seconds. Deliberately measured in wall time
# rather than frames: the whole point of it is to show how slow a frame is, and
# a frame-counted line goes quiet exactly when the loop is slowest.
PERF_REPORT_EVERY_S = 2.0


@dataclass
class FrameOutcome:
    """What the pipeline made of one frame."""

    detections: list[dict]
    status: str
    bbox: tuple[int, int, int, int] | None = None
    center: tuple[int, int] | None = None
    dx: int | None = None
    dy: int | None = None
    # How much of the frame's height the locked box fills, 0..100. This is the
    # car's only distance cue, so it is derived from `bbox` -- the box of the
    # target actually being tracked -- and never from any other detection in
    # the frame. 0 means "no locked box", which the firmware reads as "no
    # distance information" and refuses to drive forward on.
    size_pct: int = 0
    # True while the target's position is predicted rather than detected.
    coasting: bool = False


class TargetNameSender:
    """
    Sends the LCD target name -- on connection and on change, never per frame.

    The name is metadata: the STM32 latches it and prints it, so re-sending
    it every frame would spend bandwidth the tracking stream needs without
    telling the firmware anything new. Remembering the last name sent is
    what keeps it to one write per target.
    """

    def __init__(self) -> None:
        self._last_sent: str | None = None

    def forget(self) -> None:
        """Drop the memory of what was sent, so the next send goes out again."""
        self._last_sent = None

    def send(self, ser, name: str) -> bool:
        """Send `name` unless it is already on the display. True if written."""
        if ser is None:
            return False

        display = normalize_target_name(name)
        if display == self._last_sent:
            return False

        packet = encode_target_name_packet(display)
        try:
            ser.write(packet)
        except Exception as exc:
            print(f"SERIAL WRITE ERROR (target name): {exc!r}")
            self.forget()
            return False

        self._last_sent = display
        # Logged from the packet, so a name the display had to cut is
        # reported as what actually went out rather than what was asked for.
        print(f"TX target name: {packet[3:-1].decode('ascii')}")
        return True


def size_pct_from_bbox(bbox: tuple[int, int, int, int] | None, frame_height: int) -> int:
    """
    How much of the frame's height `bbox` fills, as a percentage clamped to
    0..100 -- the firmware's TARGET_SIZE_PCT, and the only distance cue the
    car has. A bigger box means a closer target.

    Height rather than area or width because it is the dimension that degrades
    most gracefully: a person walking towards the camera keeps roughly the same
    aspect ratio, while a partly cropped or turned target changes width far
    more than height.

    Returns 0 for no box and for a frame height that cannot be measured, which
    is the value that tells the firmware it has no distance information at all.
    """
    if bbox is None or frame_height <= 0:
        return 0
    _, y1, _, y2 = bbox
    return clamp_size_pct(round(100 * (y2 - y1) / frame_height))


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
        # Computed here, from this frame and this frame's tracked box, so the
        # size can never be paired with a different frame's centre error.
        size_pct=size_pct_from_bbox(result.bbox, frame.shape[0]),
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


def track_until_stopped(
    cam, detector, ser, target_class, target_color, show_display, verbose=False, name_sender=None
) -> None:
    """Follow one target until the user stops it or the camera gives out."""
    label = f"{target_color} {target_class}" if target_color else target_class
    print(f"\nTarget: {label}")
    print("Press 'q' in the window to stop." if show_display else "Press Ctrl+C to stop.")

    # One write per target, here rather than in the loop below.
    if name_sender is not None:
        name_sender.send(ser, label)

    tracker = Tracker(frame_width=cam.width, frame_height=cam.height)
    last_status = None
    dropped_frames = 0
    consecutive_detections = 0
    processed_frames = 0
    serial_rx_buffer = b""
    serial_response_received = False
    serial_warning_printed = False
    serial_wait_started = time.monotonic()
    # Timing window. `perf_frames` counts frames actually processed, so a run
    # of dropped camera frames shows up as a longer interval rather than
    # quietly flattering the average.
    perf_window_started = time.monotonic()
    perf_frames = 0
    perf_infer_ms = 0.0
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

            # The whole detect -> select -> track path for one frame. Wrapped
            # because a crash in here is otherwise a bare traceback at the
            # bottom of a fast-scrolling log: this prints the exact file and
            # line, then stops the loop so the car is not left driving on the
            # last packet it managed to send. The STM32's own stale-packet
            # failsafe stops the wheels within FAILSAFE_TIMEOUT_MS either way.
            try:
                outcome = process_frame(frame, detector, tracker, target_class, target_color)
            except Exception:
                print("FRAME PROCESSING FAILED -- exact location below:")
                traceback.print_exc()
                break
            processed_frames += 1
            perf_frames += 1
            perf_infer_ms += detector.last_inference_ms

            # What the status byte reports is what the detector confirmed on
            # *this* frame, so a coasting outcome counts as a miss: the lock is
            # still held, but the position is carried forward on velocity and
            # nothing was actually seen. Taking status == LOCKED alone would let
            # dead-reckoned frames build a "confident lock" out of stale data.
            detected = outcome.status == STATUS_LOCKED and not outcome.coasting
            consecutive_detections = consecutive_detections + 1 if detected else 0
            # DETECTED asserts on the first fresh detection (blue LED, and the
            # firmware's authority to move); LOCKED follows one fresh frame
            # later (green LED). A frame without a fresh detection zeroes
            # consecutive_detections above, which clears both immediately --
            # no coasting/stale frame ever counts as fresh.
            locked = consecutive_detections >= LOCK_FRAMES
            status = (STATUS_BIT_DETECTED if detected else 0x00) | (
                STATUS_BIT_LOCKED if locked else 0x00
            )

            if processed_frames % STATUS_DEBUG_EVERY == 0:
                link = "" if ser is not None else "  [NO SERIAL LINK -- 0 bytes sent]"
                print(
                    f"detected={detected} consecutive={consecutive_detections} "
                    f"locked={locked} status=0x{status:02X} "
                    f"size_pct={outcome.size_pct}{link}"
                )

            # Printing per frame is not free: at 30 fps a few lines a frame is
            # enough terminal I/O to cut the frame rate, and a slower loop means
            # the target moves further between detections -- the tracker's
            # hardest case. Off unless asked for.
            if verbose and outcome.status == STATUS_LOCKED:
                print(f"err_x={outcome.dx}, err_y={outcome.dy}")

            if ser is not None:
                has_fix = outcome.status == STATUS_LOCKED
                err_x = int(outcome.dx if has_fix else 0)
                err_y = int(outcome.dy if has_fix else 0)
                # Recomputed from this frame every time and sent on every
                # frame, so the firmware's copy is replaced rather than left
                # holding an old one. Without a fix it is 0, which is what
                # stops the car driving at a target that is no longer there.
                size_pct = outcome.size_pct if has_fix else 0
                packet = encode_tracking_packet(err_x, err_y, status, size_pct)
                if verbose:
                    print(
                        f"VISION TX: err_x={err_x} err_y={err_y} "
                        f"status=0x{status:02X} size_pct={size_pct} "
                        f"bytes={packet.hex(' ')}"
                    )
                try:
                    sent = ser.write(packet)
                    if verbose:
                        print(f"SENT {sent} BYTES")
                except Exception as exc:
                    print(f"SERIAL WRITE ERROR: {exc!r}")
                    ser.close()
                    ser = None
                    # The LCD's name went down with the link; whatever opens
                    # the port next has to send it again.
                    if name_sender is not None:
                        name_sender.forget()

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

            # One timing line every couple of seconds -- enough to see what the
            # loop is really doing, far too rare to slow it down. `interval` is
            # what the firmware's stale-packet failsafe has to tolerate, so
            # read it against FAILSAFE_TIMEOUT_MS before changing either.
            perf_elapsed = time.monotonic() - perf_window_started
            if perf_elapsed >= PERF_REPORT_EVERY_S and perf_frames > 0:
                print(
                    f"PERF infer={perf_infer_ms / perf_frames:.0f}ms "
                    f"interval={1000 * perf_elapsed / perf_frames:.0f}ms "
                    f"fps={perf_frames / perf_elapsed:.1f}"
                )
                perf_window_started = time.monotonic()
                perf_frames = 0
                perf_infer_ms = 0.0

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
            print(
                f"!!! NO LINK TO THE STM32 on {args.port}: no packets will be sent, so"
                " the LEDs, pan/tilt and motors will NOT respond."
            )
            print(
                "!!! Vision still reports detected/locked/status normally -- that is"
                " the vision half only. Check the port name (ls /dev/ttyACM* /dev/ttyUSB*)."
            )

    name_sender = TargetNameSender()
    show_display = not args.no_display
    print(
        f"Model on {detector.device}, camera {cam.width}x{cam.height}, "
        f"serial {'-> ' + args.port if ser is not None else 'disabled'}."
    )

    try:
        if args.command is not None:
            target = resolve_target(args.command, detector)
            if target is not None:
                track_until_stopped(
                    cam, detector, ser, *target, show_display, args.verbose, name_sender
                )
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
                track_until_stopped(
                    cam, detector, ser, *target, show_display, args.verbose, name_sender
                )
    finally:
        print("Shutting down.")
        cam.release()
        if ser is not None:
            ser.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
