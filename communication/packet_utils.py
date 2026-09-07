"""
Packet encoding/decoding utilities.

Defines the wire format used between the vision system and the STM32 and
provides helpers to pack/unpack it. One fixed-size 8-byte packet:

    byte 0     0xAA            sync
    byte 1     0x55            sync
    bytes 2-3  dx, int16 LE    horizontal center error, pixels
    bytes 4-5  dy, int16 LE    vertical center error, pixels
    byte 6     flags, uint8    bit 0 = target locked
    byte 7     checksum, uint8 XOR of bytes 2-6

Fixed size and sync-prefixed so the firmware can resynchronise mid-stream
after dropped bytes, and checksummed so corrupted frames are discarded
rather than steering the servos.
"""

import struct
from dataclasses import dataclass

SYNC_BYTES = b"\xaa\x55"
PAYLOAD_FORMAT = "<hhB"  # dx, dy, flags
PAYLOAD_SIZE = struct.calcsize(PAYLOAD_FORMAT)
CHECKSUM_SIZE = 1
PACKET_SIZE = len(SYNC_BYTES) + PAYLOAD_SIZE + CHECKSUM_SIZE

FLAG_TARGET_LOCKED = 0x01

INT16_MIN = -32768
INT16_MAX = 32767


@dataclass
class Packet:
    """One decoded center-error packet."""

    dx: int
    dy: int
    locked: bool


def _checksum(payload: bytes) -> int:
    """XOR of every payload byte."""
    checksum = 0
    for byte in payload:
        checksum ^= byte
    return checksum


def _clamp_int16(value: int) -> int:
    """Keep a pixel error inside the range the wire format can carry."""
    return max(INT16_MIN, min(INT16_MAX, int(value)))


def encode_packet(dx: int, dy: int, locked: bool) -> bytes:
    """Build the on-wire bytes for one center-error reading."""
    flags = FLAG_TARGET_LOCKED if locked else 0
    payload = struct.pack(PAYLOAD_FORMAT, _clamp_int16(dx), _clamp_int16(dy), flags)
    return SYNC_BYTES + payload + bytes([_checksum(payload)])


def decode_packet(data: bytes) -> Packet | None:
    """Decode exactly one packet, or None if it is malformed or corrupt."""
    if len(data) != PACKET_SIZE or not data.startswith(SYNC_BYTES):
        return None

    payload = data[len(SYNC_BYTES) : len(SYNC_BYTES) + PAYLOAD_SIZE]
    if _checksum(payload) != data[-1]:
        return None

    dx, dy, flags = struct.unpack(PAYLOAD_FORMAT, payload)
    return Packet(dx=dx, dy=dy, locked=bool(flags & FLAG_TARGET_LOCKED))


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
        if len(buffer) - sync_at < PACKET_SIZE:
            return None, buffer[sync_at:]

        candidate = buffer[sync_at : sync_at + PACKET_SIZE]
        packet = decode_packet(candidate)
        if packet is not None:
            return packet, buffer[sync_at + PACKET_SIZE :]
        start = sync_at + 1
