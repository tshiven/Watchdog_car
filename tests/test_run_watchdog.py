"""Tests for scripts/run_watchdog.py, driving the pipeline with a fake detector."""

import struct

import numpy as np

from communication.packet_utils import (
    SYNC_BYTES,
    TARGET_NAME_SYNC_BYTES,
    encode_target_name_packet,
)
from scripts.run_watchdog import (
    STATUS_LOCKED,
    STATUS_LOST,
    STATUS_SEARCHING,
    TargetNameSender,
    crop_to_bbox,
    process_frame,
    track_until_stopped,
)
from vision.tracker import Tracker

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ORANGE_BGR = (0, 140, 255)
GRAY_BGR = (128, 128, 128)


class FakeDetector:
    """Returns a canned list of detections, ignoring the frame."""

    def __init__(self, *frames_of_detections: list[dict]) -> None:
        self._frames = list(frames_of_detections)

    def detect(self, frame, target_class=None) -> list[dict]:
        return self._frames.pop(0) if self._frames else []


def _detection(bbox, class_name="cat", confidence=0.9) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "class_id": 15,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": bbox,
        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        "width": x2 - x1,
        "height": y2 - y1,
        "area": (x2 - x1) * (y2 - y1),
    }


def _blank_frame() -> np.ndarray:
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def _tracker() -> Tracker:
    return Tracker(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)


def test_no_detections_reports_searching():
    outcome = process_frame(_blank_frame(), FakeDetector([]), _tracker(), "cat", None)

    assert outcome.status == STATUS_SEARCHING
    assert outcome.bbox is None


def test_matching_detection_locks_on_and_reports_center_error():
    detection = _detection((100, 100, 200, 200))

    outcome = process_frame(_blank_frame(), FakeDetector([detection]), _tracker(), "cat", None)

    assert outcome.status == STATUS_LOCKED
    assert outcome.bbox == (100, 100, 200, 200)
    assert outcome.dx == 150 - FRAME_WIDTH // 2
    assert outcome.dy == 150 - FRAME_HEIGHT // 2


def test_detection_of_another_class_is_ignored():
    outcome = process_frame(
        _blank_frame(), FakeDetector([_detection((100, 100, 200, 200), "dog")]), _tracker(), "cat", None
    )

    assert outcome.status == STATUS_SEARCHING


def test_low_confidence_detection_is_not_locked_onto():
    weak = _detection((100, 100, 200, 200), confidence=0.2)

    outcome = process_frame(_blank_frame(), FakeDetector([weak]), _tracker(), "cat", None)

    assert outcome.status == STATUS_SEARCHING


def test_requested_color_decides_between_two_candidates():
    frame = _blank_frame()
    frame[100:200, 100:200] = ORANGE_BGR
    frame[100:200, 400:500] = GRAY_BGR
    orange_bbox = (100, 100, 200, 200)
    gray_bbox = (400, 100, 500, 200)
    detector = FakeDetector([_detection(gray_bbox), _detection(orange_bbox)])

    outcome = process_frame(frame, detector, _tracker(), "cat", "orange")

    assert outcome.status == STATUS_LOCKED
    assert outcome.bbox == orange_bbox


def test_target_is_followed_across_frames():
    tracker = _tracker()
    detector = FakeDetector(
        [_detection((100, 100, 200, 200))],
        [_detection((160, 105, 260, 205))],
    )
    frame = _blank_frame()

    first = process_frame(frame, detector, tracker, "cat", None)
    second = process_frame(frame, detector, tracker, "cat", None)

    assert first.status == STATUS_LOCKED
    assert second.status == STATUS_LOCKED
    assert second.bbox == (160, 105, 260, 205)
    assert second.dx > first.dx


def test_target_is_declared_lost_after_it_disappears():
    tracker = _tracker()
    frame = _blank_frame()
    process_frame(frame, FakeDetector([_detection((100, 100, 200, 200))]), tracker, "cat", None)

    statuses = [
        process_frame(frame, FakeDetector([]), tracker, "cat", None).status for _ in range(10)
    ]

    assert STATUS_LOST in statuses
    assert statuses[-1] == STATUS_SEARCHING  # falls back to searching afterwards


def test_crop_to_bbox_clips_a_box_that_runs_past_the_frame_edge():
    frame = _blank_frame()

    crop = crop_to_bbox(frame, (-50, -50, 100, 100))

    assert crop.shape[0] == 100
    assert crop.shape[1] == 100


def _hsv_bgr(hue: int, saturation: int = 200, value: int = 210):
    import cv2

    return cv2.cvtColor(
        np.full((1, 1, 3), (hue, saturation, value), dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0]


def _frame_with_object(bbox, color) -> np.ndarray:
    frame = _blank_frame()
    x1, y1, x2, y2 = bbox
    frame[y1:y2, x1:x2] = color
    return frame


def test_lock_survives_a_frame_where_the_color_score_dips():
    # The colour filter used to run on every frame and hide the target from
    # the tracker whenever its score dipped -- a highlight, a shadow, a hand
    # passing over it. A few such frames in a row ended the lock on an object
    # that had not gone anywhere. Colour now decides only what to lock onto.
    tracker = _tracker()
    orange = _hsv_bgr(15)

    lit = _frame_with_object((100, 100, 200, 200), orange)
    first = process_frame(lit, FakeDetector([_detection((100, 100, 200, 200))]),
                          tracker, "cat", "orange")
    assert first.status == STATUS_LOCKED

    # Same object, now washed out by a highlight: barely orange any more.
    washed = _frame_with_object((110, 100, 210, 200), _hsv_bgr(15, 20, 250))
    second = process_frame(washed, FakeDetector([_detection((110, 100, 210, 200))]),
                           tracker, "cat", "orange")

    assert second.status == STATUS_LOCKED
    assert second.bbox == (110, 100, 210, 200)


def test_a_wrongly_colored_object_is_not_locked_onto_in_the_first_place():
    frame = _frame_with_object((100, 100, 200, 200), _hsv_bgr(110))  # blue

    outcome = process_frame(frame, FakeDetector([_detection((100, 100, 200, 200))]),
                            _tracker(), "cat", "orange")

    assert outcome.status == STATUS_SEARCHING


def test_class_matching_is_case_insensitive_end_to_end():
    detection = _detection((100, 100, 200, 200), class_name="Cat")

    outcome = process_frame(_blank_frame(), FakeDetector([detection]),
                            _tracker(), "cat", None)

    assert outcome.status == STATUS_LOCKED


def test_a_mid_confidence_bottle_is_locked_onto():
    detection = _detection((100, 100, 200, 200), class_name="bottle", confidence=0.42)

    outcome = process_frame(_blank_frame(), FakeDetector([detection]),
                            _tracker(), "bottle", None)

    assert outcome.status == STATUS_LOCKED


def test_a_fast_small_target_is_followed_across_frames():
    tracker = Tracker(frame_width=1280, frame_height=720)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def bottle(x):
        return _detection((x, 300, x + 60, 460), class_name="bottle", confidence=0.45)

    x = 600
    outcome = process_frame(frame, FakeDetector([bottle(x)]), tracker, "bottle", None)
    assert outcome.status == STATUS_LOCKED

    for _ in range(4):
        x += 140
        outcome = process_frame(frame, FakeDetector([bottle(x)]), tracker, "bottle", None)

        assert outcome.status == STATUS_LOCKED
        assert outcome.bbox == (x, 300, x + 60, 460)


def test_coasting_is_reported_on_the_outcome():
    tracker = _tracker()
    process_frame(_blank_frame(), FakeDetector([_detection((100, 100, 200, 200))]),
                  tracker, "cat", None)

    outcome = process_frame(_blank_frame(), FakeDetector([]), tracker, "cat", None)

    assert outcome.status == STATUS_LOCKED
    assert outcome.coasting


# --- Target-name transmission -------------------------------------------


class FakeSerial:
    """Stands in for pyserial's Serial: records writes, never has input."""

    def __init__(self, fail_on_write: bool = False) -> None:
        self.written = b""
        self.closed = False
        self.in_waiting = 0
        self._fail_on_write = fail_on_write

    def write(self, data: bytes) -> int:
        if self._fail_on_write:
            raise OSError("port went away")
        self.written += data
        return len(data)

    def readline(self) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


class FakeCamera:
    """Yields canned frames, then stops the loop the way Ctrl+C would."""

    def __init__(self, frames: int) -> None:
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        self._remaining = frames

    def read(self):
        if self._remaining <= 0:
            raise KeyboardInterrupt
        self._remaining -= 1
        return _blank_frame()


def _name_packets(written: bytes) -> list[bytes]:
    """
    Every target-name frame in a recorded write stream.

    Both message types are <2-byte header><LEN><payload><checksum>, so one
    walk covers the stream; the header says which is which.
    """
    packets, index = [], 0
    while index < len(written):
        header = written[index : index + 2]
        assert header in (SYNC_BYTES, TARGET_NAME_SYNC_BYTES), "unknown frame header"
        frame = written[index : index + 4 + written[index + 2]]
        if header == TARGET_NAME_SYNC_BYTES:
            packets.append(frame)
        index += len(frame)
    return packets


def test_target_name_is_sent_once_when_tracking_starts():
    fake = FakeSerial()
    sender = TargetNameSender()

    sender.send(fake, "orange cat")

    assert fake.written == encode_target_name_packet("ORANGE CAT")


def test_target_name_is_not_resent_while_the_target_is_unchanged():
    fake = FakeSerial()
    sender = TargetNameSender()

    assert sender.send(fake, "orange cat") is True
    assert sender.send(fake, "orange cat") is False
    assert sender.send(fake, "  ORANGE   cat ") is False  # same after normalising

    assert len(fake.written) == len(encode_target_name_packet("ORANGE CAT"))


def test_target_name_is_resent_when_the_target_changes():
    fake = FakeSerial()
    sender = TargetNameSender()

    sender.send(fake, "orange cat")
    sender.send(fake, "red bottle")

    assert _name_packets(fake.written) == [
        encode_target_name_packet("ORANGE CAT"),
        encode_target_name_packet("RED BOTTLE"),
    ]


def test_target_name_is_resent_after_a_reconnect():
    sender = TargetNameSender()
    first = FakeSerial()
    sender.send(first, "orange cat")

    sender.forget()  # what a dropped link does
    reconnected = FakeSerial()
    sender.send(reconnected, "orange cat")

    assert reconnected.written == encode_target_name_packet("ORANGE CAT")


def test_target_name_send_without_a_serial_link_is_a_no_op():
    assert TargetNameSender().send(None, "orange cat") is False


def test_a_failed_name_write_is_survived_and_retried_later():
    sender = TargetNameSender()

    assert sender.send(FakeSerial(fail_on_write=True), "orange cat") is False

    working = FakeSerial()
    assert sender.send(working, "orange cat") is True
    assert working.written == encode_target_name_packet("ORANGE CAT")


def test_tracking_packets_are_unchanged_and_follow_the_name_packet():
    detection = _detection((100, 100, 200, 200))
    detector = FakeDetector([detection], [detection])
    fake = FakeSerial()

    track_until_stopped(
        FakeCamera(frames=2),
        detector,
        fake,
        "cat",
        None,
        show_display=False,
        name_sender=TargetNameSender(),
    )

    name_packet = encode_target_name_packet("CAT")
    assert fake.written.startswith(name_packet)
    assert fake.written[:2] == TARGET_NAME_SYNC_BYTES
    assert name_packet == b"\xa5\x5a\x03CAT" + bytes([0x03 ^ 0x43 ^ 0x41 ^ 0x54])

    # Exactly one name frame, whatever the frame count.
    assert len(_name_packets(fake.written)) == 1

    # Everything after it is the untouched 9-byte tracking frame.
    tracking = fake.written[len(name_packet) :]
    assert len(tracking) == 2 * 9
    for offset in (0, 9):
        frame = tracking[offset : offset + 9]
        assert frame[:3] == b"\xaa\x55\x05"
        err_x, err_y, status = struct.unpack("<hhB", frame[3:8])
        checksum = 0x05
        for byte in frame[3:8]:
            checksum ^= byte
        assert frame[8] == checksum
        # The box centres at (150, 150) in a 640x480 frame.
        assert (err_x, err_y) == (150 - FRAME_WIDTH // 2, 150 - FRAME_HEIGHT // 2)
        assert status in (0x00, 0x01)  # two frames is far short of a lock
