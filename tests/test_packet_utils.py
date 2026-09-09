"""Tests for communication/packet_utils.py."""

from communication.packet_utils import (
    INT16_MAX,
    INT16_MIN,
    PACKET_SIZE,
    SYNC_BYTES,
    decode_packet,
    encode_packet,
    find_packet,
)


def test_encoded_packet_has_fixed_size_and_sync_prefix():
    data = encode_packet(10, -20)

    assert len(data) == PACKET_SIZE
    assert data.startswith(SYNC_BYTES)
    assert data[2] == 4


def test_round_trip_preserves_values():
    packet = decode_packet(encode_packet(-150, 275))

    assert packet is not None
    assert packet.dx == -150
    assert packet.dy == 275


def test_decode_rejects_corrupted_checksum():
    data = bytearray(encode_packet(10, 20))
    data[-1] ^= 0xFF

    assert decode_packet(bytes(data)) is None


def test_decode_rejects_wrong_length():
    data = bytearray(encode_packet(1, 2))
    data[2] = 3
    assert decode_packet(bytes(data)) is None


def test_decode_rejects_missing_sync():
    assert decode_packet(b"\x00" * PACKET_SIZE) is None


def test_values_are_clamped_to_int16_range():
    packet = decode_packet(encode_packet(999999, -999999))

    assert packet is not None
    assert packet.dx == INT16_MAX
    assert packet.dy == INT16_MIN


def test_find_packet_skips_leading_garbage():
    stream = b"\x01\x02\x03" + encode_packet(5, 6)

    packet, rest = find_packet(stream)

    assert packet is not None
    assert (packet.dx, packet.dy) == (5, 6)
    assert rest == b""


def test_find_packet_returns_remaining_bytes():
    stream = encode_packet(1, 2) + encode_packet(3, 4)

    first, rest = find_packet(stream)
    second, remainder = find_packet(rest)

    assert first is not None and (first.dx, first.dy) == (1, 2)
    assert second is not None and (second.dx, second.dy) == (3, 4)
    assert remainder == b""


def test_find_packet_buffers_incomplete_packet():
    partial = encode_packet(7, 8)[:-2]

    packet, rest = find_packet(partial)

    assert packet is None
    assert rest == partial


def test_find_packet_recovers_after_corrupt_packet():
    corrupt = bytearray(encode_packet(1, 1))
    corrupt[-1] ^= 0xFF
    stream = bytes(corrupt) + encode_packet(42, 43)

    packet, _ = find_packet(stream)

    assert packet is not None
    assert (packet.dx, packet.dy) == (42, 43)
