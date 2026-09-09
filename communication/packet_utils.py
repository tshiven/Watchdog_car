"""
Packet encoding/decoding utilities.

Defines the wire format used between the vision system and the STM32 and
provides helpers to pack/unpack it. One fixed-size 8-byte frame:

    byte 0     0xAA            sync
    byte 1     0x55            sync
    byte 2     0x04            payload length
    bytes 3-4  dx, int16 LE    horizontal center error, pixels
    bytes 5-6  dy, int16 LE    vertical center error, pixels
    byte 7     checksum, uint8 XOR of byte 2 and bytes 3-6

Fixed size and sync-prefixed so the firmware can resynchronise mid-stream
after dropped bytes, and checksummed so corrupted frames are discarded
rather than steering the servos.
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
