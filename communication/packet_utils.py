"""
Packet encoding/decoding utilities.

Defines the wire format used between the vision system and the STM32 and
provides helpers to pack/unpack it. The center-error frame is a fixed
8 bytes:

    byte 0     0xAA            sync
    byte 1     0x55            sync
    byte 2     0x04            payload length
    bytes 3-4  dx, int16 LE    horizontal center error, pixels
    bytes 5-6  dy, int16 LE    vertical center error, pixels
    byte 7     checksum, uint8 XOR of byte 2 and bytes 3-6

Fixed size and sync-prefixed so the firmware can resynchronise mid-stream
after dropped bytes, and checksummed so corrupted frames are discarded
rather than steering the servos.

A second, low-rate frame carries the target name for the STM32's LCD; see
the target-name section at the bottom of this file.
"""

import struct
from dataclasses import dataclass

SYNC_BYTES = b"\xaa\x55"
PAYLOAD_FORMAT = "<hh"  # err_x, err_y
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FORMAT)
LENGTH_BYTE = PAYLOAD_SIZE
CHECKSUM_SIZE = 1
FRAME_SIZE = len(SYNC_BYTES) + 1 + PAYLOAD_SIZE + CHECKSUM_SIZE

PACKET_SIZE = FRAME_SIZE

INT16_MIN = -32768
INT16_MAX = 32767


@dataclass
class Packet:
    """One decoded center-error packet."""

    dx: int
    dy: int


def _checksum(length: int, payload: bytes) -> int:
    """XOR the payload length and every payload byte."""
    checksum = length
    for byte in payload:
        checksum ^= byte
    return checksum


def _clamp_int16(value: int) -> int:
    """Keep a pixel error inside the range the wire format can carry."""
    return max(INT16_MIN, min(INT16_MAX, int(value)))


def encode_packet(err_x: int, err_y: int) -> bytes:
    """Build the on-wire bytes for one center-error reading."""
    err_x = _clamp_int16(err_x)
    err_y = _clamp_int16(err_y)
    payload = struct.pack("<hh", err_x, err_y)
    return SYNC_BYTES + bytes([LENGTH_BYTE]) + payload + bytes([_checksum(LENGTH_BYTE, payload)])


def decode_packet(data: bytes) -> Packet | None:
    """Decode exactly one packet, or None if it is malformed or corrupt."""
    if len(data) != FRAME_SIZE or not data.startswith(SYNC_BYTES):
        return None

    length = data[len(SYNC_BYTES)]
    if length != LENGTH_BYTE:
        return None
    payload_start = len(SYNC_BYTES) + 1
    payload = data[payload_start : payload_start + PAYLOAD_SIZE]
    if _checksum(length, payload) != data[-1]:
        return None

    dx, dy = struct.unpack(PAYLOAD_FORMAT, payload)
    return Packet(dx=dx, dy=dy)


def find_packet(buffer: bytes) -> tuple[Packet | None, bytes]:
    """
    Pull the first valid packet out of a serial byte stream.

    Returns the packet (or None if there isn't a complete valid one yet) and
    the bytes still left to process. Leading garbage and packets that fail
    their checksum are skipped, so a corrupted stream recovers on its own.
    """
    start = 0
    while True:
        sync_at = buffer.find(SYNC_BYTES, start)
        if sync_at == -1:
            # Keep a trailing byte back: it may be the first half of a sync pair.
            return None, buffer[-1:] if buffer.endswith(SYNC_BYTES[:1]) else b""
        if len(buffer) - sync_at < FRAME_SIZE:
            return None, buffer[sync_at:]

        candidate = buffer[sync_at : sync_at + FRAME_SIZE]
        packet = decode_packet(candidate)
        if packet is not None:
            return packet, buffer[sync_at + FRAME_SIZE :]
        start = sync_at + 1


# --- Target-name packet -------------------------------------------------
#
# A separate, low-rate message that tells the STM32 what to print on the
# first line of its 16x2 LCD. It has its own header and a variable-length
# ASCII payload:
#
#     byte 0       0xA5             sync
#     byte 1       0x5A             sync
#     byte 2       LEN              payload length, 0..16
#     bytes 3..    name, ASCII      exactly LEN bytes, unpadded
#     last byte    checksum, uint8  XOR of LEN and every payload byte
#
# The header is what tells the two message types apart -- 0xA5 0x5A here
# against 0xAA 0x55 for the center-error frame -- so the length byte is free
# to be the real name length. No padding and no terminator: the firmware
# reads exactly LEN bytes.

TARGET_NAME_SYNC_BYTES = b"\xa5\x5a"
TARGET_NAME_MAX_CHARS = 16

# Anything outside printable ASCII becomes this: the LCD's character ROM has
# no glyph for it, and a replacement is easier to read than a stray byte.
TARGET_NAME_SUBSTITUTE = "?"


def normalize_target_name(name: str) -> str:
    """
    Turn a user-facing target label into what the LCD should show.

    Trims the ends, folds every run of whitespace (tabs and newlines
    included, which would otherwise garble the display) into one space,
    uppercases it to match the LCD's look, and replaces characters the
    display cannot render. Truncation is left to the encoder, which has to
    enforce the length anyway.
    """
    collapsed = " ".join(str(name).split())
    rendered = "".join(
        char if " " <= char <= "~" else TARGET_NAME_SUBSTITUTE for char in collapsed
    )
    return rendered.upper()


def encode_target_name_packet(name: str) -> bytes:
    """
    Build the on-wire bytes for one target-name message.

    Accepts any string: it is normalised for the display, forced to ASCII and
    cut to the 16 characters the LCD holds. The frame carries the resulting
    length, so it is 4 + LEN bytes rather than a fixed size.
    """
    payload = (
        normalize_target_name(name)
        .encode("ascii", errors="replace")[:TARGET_NAME_MAX_CHARS]
        # Only a truncation can leave a trailing space, and a space the
        # display would not show is a byte the firmware need not read.
        .rstrip(b" ")
    )
    length = len(payload)
    return (
        TARGET_NAME_SYNC_BYTES
        + bytes([length])
        + payload
        + bytes([_checksum(length, payload)])
    )
