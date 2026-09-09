"""
Serial communication protocol module.

Sends center-error packets (see packet_utils.py) to the STM32 over a serial
link and decodes whatever it sends back. Reads are non-blocking: poll()
drains the bytes currently waiting and returns only the complete, valid
packets found, buffering any partial packet until the rest arrives.

pyserial is imported lazily so the packet layer stays importable (and
testable) on machines without it installed.
"""

if __name__ == "__main__":  # Allow running this file directly, not just with -m.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from communication.packet_utils import Packet, encode_packet, find_packet

DEFAULT_BAUDRATE = 115200
DEFAULT_TIMEOUT_S = 0.1


class SerialError(RuntimeError):
    """Raised when the serial port cannot be opened."""


class SerialLink:
    """A serial connection to the STM32, speaking the Watchdog packet format."""

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = DEFAULT_TIMEOUT_S,
        transport=None,
    ) -> None:
        self.port = port
        self._serial = transport if transport is not None else _open_port(port, baudrate, timeout)
        self._buffer = b""

    def send_center_error(self, dx: int, dy: int, locked: bool | None = None) -> None:
        """Send one center-error reading to the STM32."""
        packet = encode_packet(dx, dy)
        self._serial.write(packet)
        print(
            f"TX err_x={max(-32768, min(32767, int(dx)))} "
            f"err_y={max(-32768, min(32767, int(dy)))} "
            f"bytes={' '.join(f'{byte:02X}' for byte in packet)}"
        )

    def poll(self) -> list[Packet]:
        """Return every complete packet available right now; never blocks."""
        pending = self._serial.in_waiting
        if pending:
            self._buffer += self._serial.read(pending)

        packets = []
        while True:
            packet, self._buffer = find_packet(self._buffer)
            if packet is None:
                return packets
            packets.append(packet)

    def close(self) -> None:
        self._serial.close()

    def __enter__(self) -> "SerialLink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _open_port(port: str, baudrate: int, timeout: float):
    """Open a pyserial port, converting any failure into SerialError."""
    try:
        import serial
    except ImportError as exc:
        raise SerialError("pyserial is not installed; run 'pip install pyserial'") from exc

    try:
        return serial.Serial(port, baudrate, timeout=timeout)
    except Exception as exc:
        raise SerialError(f"Could not open serial port {port}: {exc}") from exc


def list_ports() -> list[str]:
    """Device names of the serial ports currently attached, for finding the STM32."""
    try:
        from serial.tools import list_ports as pyserial_list_ports
    except ImportError as exc:
        raise SerialError("pyserial is not installed; run 'pip install pyserial'") from exc
    return [info.device for info in pyserial_list_ports.comports()]


def _run_demo() -> None:
    """Print available ports, or stream test packets to one."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Serial link demo")
    parser.add_argument("--port", default=None, help="Serial port to send test packets to")
    parser.add_argument("--list", action="store_true", help="List serial ports and exit")
    args = parser.parse_args()

    if args.list or args.port is None:
        print(f"Available serial ports: {list_ports()}")
        return

    try:
        link = SerialLink(args.port)
    except SerialError as exc:
        print(exc)
        return

    print(f"Sending test packets to {args.port}. Press Ctrl+C to stop.")
    try:
        for dx in range(-100, 101, 10):
            link.send_center_error(dx, -dx)
            for packet in link.poll():
                print(f"Received: {packet}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()


if __name__ == "__main__":
    _run_demo()
