"""Tests for communication/serial_protocol.py, against a fake serial port."""

from communication.packet_utils import decode_packet, encode_packet
from communication.serial_protocol import SerialLink


class FakeSerial:
    """Stands in for pyserial's Serial: records writes, replays queued reads."""

    def __init__(self, to_read: bytes = b"") -> None:
        self.written = b""
        self._to_read = to_read
        self.closed = False

    @property
    def in_waiting(self) -> int:
        return len(self._to_read)

    def read(self, size: int) -> bytes:
        chunk, self._to_read = self._to_read[:size], self._to_read[size:]
        return chunk

    def write(self, data: bytes) -> int:
        self.written += data
        return len(data)

    def close(self) -> None:
        self.closed = True

    def queue(self, data: bytes) -> None:
        self._to_read += data


def test_send_center_error_writes_encoded_packet():
    fake = FakeSerial()
    link = SerialLink("fake", transport=fake)

    link.send_center_error(-30, 45, locked=True)

    packet = decode_packet(fake.written)
    assert packet is not None
    assert (packet.dx, packet.dy, packet.locked) == (-30, 45, True)


def test_poll_returns_nothing_when_port_is_quiet():
    link = SerialLink("fake", transport=FakeSerial())

    assert link.poll() == []


def test_poll_decodes_multiple_packets():
    fake = FakeSerial(encode_packet(1, 2, locked=True) + encode_packet(3, 4, locked=False))
    link = SerialLink("fake", transport=fake)

    packets = link.poll()

    assert [(p.dx, p.dy) for p in packets] == [(1, 2), (3, 4)]


def test_poll_reassembles_packet_split_across_reads():
    data = encode_packet(11, 22, locked=True)
    fake = FakeSerial(data[:3])
    link = SerialLink("fake", transport=fake)

    assert link.poll() == []

    fake.queue(data[3:])
    packets = link.poll()

    assert len(packets) == 1
    assert (packets[0].dx, packets[0].dy) == (11, 22)


def test_close_closes_underlying_port():
    fake = FakeSerial()

    with SerialLink("fake", transport=fake):
        pass

    assert fake.closed
