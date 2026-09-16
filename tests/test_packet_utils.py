"""Tests for communication/packet_utils.py."""

from communication.packet_utils import (
    INT16_MAX,
    INT16_MIN,
    PACKET_SIZE,
    SYNC_BYTES,
    TARGET_NAME_MAX_CHARS,
    TARGET_NAME_SYNC_BYTES,
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
    """The ASCII bytes between the length byte and the checksum."""
    return packet[3:-1]


def test_target_name_packet_uses_its_own_header():
    packet = encode_target_name_packet("ORANGE CAT")

    assert packet[:2] == TARGET_NAME_SYNC_BYTES
    assert packet[:2] == b"\xa5\x5a"


def test_target_name_header_is_not_the_tracking_header():
    packet = encode_target_name_packet("A CAT")

    assert packet[:2] != SYNC_BYTES  # tracking is AA 55


def test_target_name_length_byte_is_the_actual_payload_length():
    packet = encode_target_name_packet("ORANGE CAT")

    assert packet[2] == 0x0A
    assert packet[2] == len("ORANGE CAT")
    assert packet[2] == len(_name_payload(packet))


def test_target_name_payload_is_the_name_with_no_padding():
    packet = encode_target_name_packet("ORANGE CAT")

    assert _name_payload(packet) == b"ORANGE CAT"
    assert b"\x00" not in packet[2:]  # no terminator
    assert not _name_payload(packet).endswith(b" ")  # no padding


def test_orange_cat_encodes_to_the_frame_the_firmware_expects():
    assert encode_target_name_packet("orange cat") == bytes(
        [0xA5, 0x5A, 0x0A, 0x4F, 0x52, 0x41, 0x4E, 0x47, 0x45, 0x20, 0x43, 0x41, 0x54, 0x6C]
    )


def test_target_name_frame_is_four_bytes_plus_the_payload():
    packet = encode_target_name_packet("ORANGE CAT")

    assert len(packet) == 4 + 10


def test_target_name_frame_size_varies_with_the_name():
    short = encode_target_name_packet("CAT")
    long = encode_target_name_packet("ORANGE CAT")

    assert len(short) == 4 + 3
    assert len(long) == 4 + 10
    assert len(short) != len(long)


def test_target_name_checksum_is_xor_of_length_and_payload():
    packet = encode_target_name_packet("ORANGE CAT")

    expected = packet[2]
    for byte in _name_payload(packet):
        expected ^= byte
    assert packet[-1] == expected


def test_target_name_longer_than_the_display_is_truncated_not_dropped():
    packet = encode_target_name_packet("A VERY LONG TARGET NAME INDEED")

    assert packet[2] == TARGET_NAME_MAX_CHARS
    assert _name_payload(packet) == b"A VERY LONG TARG"
    assert len(packet) == 4 + TARGET_NAME_MAX_CHARS


def test_truncation_never_exceeds_the_maximum_for_any_name():
    for name in ("X" * 100, "WORD " * 20, "ORANGE CAT AND A DOG"):
        packet = encode_target_name_packet(name)

        assert packet[2] <= TARGET_NAME_MAX_CHARS
        assert len(_name_payload(packet)) == packet[2]


def test_a_name_cut_mid_space_does_not_send_a_trailing_pad_byte():
    packet = encode_target_name_packet("ORANGE CAT SEEN NOW")

    assert _name_payload(packet) == b"ORANGE CAT SEEN"
    assert packet[2] == 15


def test_encoder_normalizes_the_name_it_is_given():
    assert encode_target_name_packet("  orange   cat ") == encode_target_name_packet(
        "ORANGE CAT"
    )


def test_target_name_encodes_non_ascii_safely():
    packet = encode_target_name_packet("CAFÉ CUP")

    assert all(byte < 128 for byte in _name_payload(packet))
    assert _name_payload(packet) == b"CAF? CUP"
    assert packet[2] == len(_name_payload(packet))


def test_empty_target_name_still_produces_a_well_formed_frame():
    packet = encode_target_name_packet("")

    assert packet == TARGET_NAME_SYNC_BYTES + b"\x00\x00"
    assert packet[2] == 0


def test_normalize_uppercases_and_trims():
    assert normalize_target_name("  orange cat  ") == "ORANGE CAT"


def test_normalize_collapses_inner_whitespace():
    assert normalize_target_name("orange\tcat\n") == "ORANGE CAT"


def test_normalize_replaces_characters_the_display_cannot_show():
    assert normalize_target_name("café cup") == "CAF? CUP"
