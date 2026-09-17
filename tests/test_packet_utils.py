"""Tests for communication/packet_utils.py."""

import struct

from communication.packet_utils import (
    INT16_MAX,
    INT16_MIN,
    PACKET_SIZE,
    STATUS_DETECTED,
    STATUS_LOCKED,
    SYNC_BYTES,
    TARGET_NAME_MAX_CHARS,
    TARGET_NAME_SYNC_BYTES,
    clamp_size_pct,
    decode_packet,
    encode_packet,
    encode_target_name_packet,
    encode_tracking_packet,
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


# --- Tracking packet (LEN = 6) -------------------------------------------


def test_tracking_packet_framing_and_checksum():
    packet = encode_tracking_packet(-150, 40, 0x03, 62)

    assert packet[:2] == SYNC_BYTES
    assert packet[2] == 6
    assert len(packet) == 10

    err_x, err_y, status, size_pct = struct.unpack("<hhBB", packet[3:9])
    assert (err_x, err_y, status, size_pct) == (-150, 40, 0x03, 62)

    # XOR of the length byte and every payload byte, exactly as the firmware
    # computes it.
    checksum = packet[2]
    for byte in packet[3:9]:
        checksum ^= byte
    assert packet[9] == checksum


def test_tracking_packet_uses_the_same_header_as_the_short_frame():
    """Never the target-name header: the two messages must stay distinct."""
    packet = encode_tracking_packet(0, 0, 0, 0)
    assert packet[:2] == SYNC_BYTES
    assert packet[:2] != TARGET_NAME_SYNC_BYTES


def test_tracking_packet_errors_are_signed_little_endian():
    packet = encode_tracking_packet(-2, 0, 0, 0)
    # -2 as int16 LE is FE FF; a big-endian or unsigned encoding is not.
    assert packet[3:5] == b"\xfe\xff"


def test_tracking_packet_clamps_the_size_byte():
    assert encode_tracking_packet(0, 0, 0, 250)[8] == 100
    assert encode_tracking_packet(0, 0, 0, -5)[8] == 0
    assert encode_tracking_packet(0, 0, 0, 100)[8] == 100
    assert encode_tracking_packet(0, 0, 0, 0)[8] == 0


def test_tracking_packet_clamps_the_errors_like_the_short_frame():
    packet = encode_tracking_packet(INT16_MAX + 500, INT16_MIN - 500, 0, 0)
    err_x, err_y = struct.unpack("<hh", packet[3:7])
    assert (err_x, err_y) == (INT16_MAX, INT16_MIN)


def test_clamp_size_pct_bounds():
    assert clamp_size_pct(-1) == 0
    assert clamp_size_pct(0) == 0
    assert clamp_size_pct(55) == 55
    assert clamp_size_pct(100) == 100
    assert clamp_size_pct(101) == 100


def test_status_bits_match_the_firmware():
    assert STATUS_DETECTED == 0x01
    assert STATUS_LOCKED == 0x02
