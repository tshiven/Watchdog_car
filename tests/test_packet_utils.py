"""Tests for communication/packet_utils.py."""

from communication.packet_utils import (
    INT16_MAX,
    INT16_MIN,
    PACKET_SIZE,
    SYNC_BYTES,
    TARGET_NAME_FRAME_SIZE,
    TARGET_NAME_LENGTH_BYTE,
    TARGET_NAME_MAX_CHARS,
    decode_packet,
    encode_packet,
    encode_target_name_packet,
    find_packet,
    normalize_target_name,
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


# --- Target-name packet -------------------------------------------------


def _name_payload(packet: bytes) -> bytes:
    return packet[3:-1]


def test_target_name_packet_has_fixed_size_and_length_byte():
    packet = encode_target_name_packet("ORANGE CAT")

    assert len(packet) == TARGET_NAME_FRAME_SIZE
    assert len(packet) == 20
    assert packet.startswith(SYNC_BYTES)
    assert packet[2] == 0x10


def test_target_name_payload_is_space_padded_to_sixteen():
    packet = encode_target_name_packet("ORANGE CAT")

    assert _name_payload(packet) == b"ORANGE CAT      "
    assert len(_name_payload(packet)) == TARGET_NAME_MAX_CHARS


def test_target_name_checksum_is_xor_of_length_and_payload():
    packet = encode_target_name_packet("ORANGE CAT")

    expected = TARGET_NAME_LENGTH_BYTE
    for byte in _name_payload(packet):
        expected ^= byte
    assert packet[-1] == expected


def test_target_name_longer_than_the_display_is_truncated_not_dropped():
    packet = encode_target_name_packet("A VERY LONG TARGET NAME INDEED")

    assert len(packet) == TARGET_NAME_FRAME_SIZE
    assert _name_payload(packet) == b"A VERY LONG TARG"


def test_target_name_length_byte_cannot_collide_with_a_tracking_frame():
    # The tracking frame's length byte is 5; a five-character name must not
    # produce one, or the firmware would steer on it.
    packet = encode_target_name_packet("A CAT")

    assert packet[2] == 0x10
    assert packet[2] != 0x05


def test_target_name_encodes_non_ascii_safely():
    packet = encode_target_name_packet("CAFÉ CUP")

    assert len(packet) == TARGET_NAME_FRAME_SIZE
    assert all(byte < 128 for byte in _name_payload(packet))


def test_empty_target_name_still_produces_a_well_formed_frame():
    packet = encode_target_name_packet("")

    assert len(packet) == TARGET_NAME_FRAME_SIZE
    assert _name_payload(packet) == b" " * TARGET_NAME_MAX_CHARS


def test_normalize_uppercases_and_trims():
    assert normalize_target_name("  orange cat  ") == "ORANGE CAT"


def test_normalize_collapses_inner_whitespace():
    assert normalize_target_name("orange\tcat\n") == "ORANGE CAT"


def test_normalize_replaces_characters_the_display_cannot_show():
    assert normalize_target_name("café cup") == "CAF? CUP"
