"""Upload, byte-verify and start an Arty A7 program over 115200 8N1.

Example: python host/load.py --port COM5 firmware/demo.bin
Put SW0 low and press BTN0 before downloading. Requires pyserial.
"""

import argparse
import binascii
import struct
import sys
import time
from pathlib import Path


def crc(data):
    return binascii.crc_hqx(data, 0xFFFF)


def frame(op, sequence, address, length, data=b""):
    body = struct.pack("<BBIH", op, sequence, address, length) + data
    return b"\xa5\x5a" + body + struct.pack("<H", crc(body))


class Loader:
    def __init__(self, port):
        self.port = port
        self.sequence = 0

    def read_exact(self, count):
        data = bytearray()
        deadline = time.monotonic() + 3
        while len(data) < count and time.monotonic() < deadline:
            data.extend(self.port.read(count - len(data)))
        if len(data) != count:
            raise TimeoutError(
                "Board reply timed out. Set SW0 low, press BTN0, and retry."
            )
        return bytes(data)

    def request(self, op, address=0, length=0, data=b""):
        sequence = self.sequence
        self.sequence = (sequence + 1) & 255
        self.port.write(frame(op, sequence, address, length, data))
        self.port.flush()
        header = self.read_exact(7)
        if header[:2] != b"\x5a\xa5" or header[2:4] != bytes((op | 128, sequence)):
            raise ValueError(f"Invalid response header: {header.hex()}")
        count = struct.unpack_from("<H", header, 5)[0]
        if count > 256:
            raise ValueError("Invalid response length")
        payload = self.read_exact(count)
        checksum = struct.unpack("<H", self.read_exact(2))[0]
        if crc(header[2:] + payload) != checksum:
            raise ValueError("Board response CRC mismatch")
        if header[4]:
            raise ValueError(f"Board rejected request: status={header[4]}")
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="COM5 or /dev/ttyUSB1")
    parser.add_argument("binary", type=Path)
    parser.add_argument("--console-seconds", type=float, default=30)
    args = parser.parse_args()
    import serial

    program = args.binary.read_bytes()
    if not 0 < len(program) <= 128 * 1024:
        raise ValueError("Binary must contain 1..131072 bytes linked at 0x80000000")
    with serial.Serial(
        args.port, 115200, timeout=0.1, write_timeout=3, rtscts=False, dsrdtr=False
    ) as port:
        port.reset_input_buffer()
        loader = Loader(port)
        info = loader.request(4)
        if len(info) != 12 or info[:4] != b"PA7\x01":
            raise ValueError("Board protocol/version mismatch")
        capacity, clock = struct.unpack_from("<II", info, 4)
        if len(program) > capacity:
            raise ValueError("Program exceeds board memory")
        print(f"Board RAM={capacity} clock={clock}; uploading {len(program)} bytes")
        for offset in range(0, len(program), 256):
            chunk = program[offset : offset + 256]
            loader.request(1, 0x80000000 + offset, len(chunk), chunk)
            actual = loader.request(2, 0x80000000 + offset, len(chunk))
            if actual != chunk:
                raise ValueError(
                    f"Readback mismatch at offset {offset:#x}; CPU remains stopped"
                )
        loader.request(3, 0x80000000)
        print("Readback verified. CPU started. Console:")
        deadline = time.monotonic() + args.console_seconds
        transcript = bytearray()
        while time.monotonic() < deadline:
            data = port.read(1024)
            if data:
                transcript.extend(data)
                sys.stdout.write(data.decode("ascii", errors="replace"))
                sys.stdout.flush()
        if b"FAIL " in transcript or b"TRAP " in transcript:
            raise RuntimeError("Firmware reported a failure")
        if b"PASS PEBBLE_A7_FULL_CHIP" not in transcript:
            print(
                "\nNo demo PASS observed; inspect console (custom programs may use different output)."
            )


if __name__ == "__main__":
    main()
