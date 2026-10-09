"""Host protocol checks with fragmented replies and independent known CRC."""

import importlib.util
import struct
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "loader", Path(__file__).parents[1] / "host/load.py"
)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class Port:
    def __init__(self, reply):
        self.reply = bytearray(reply)
        self.writes = bytearray()

    def write(self, value):
        self.writes.extend(value)

    def read(self, length):
        result = bytes(self.reply[: min(length, 2)])
        del self.reply[: len(result)]
        return result

    def flush(self):
        pass


def reply(op, sequence, status=0, payload=b""):
    body = (
        bytes((op | 128, sequence, status)) + struct.pack("<H", len(payload)) + payload
    )
    return b"\x5a\xa5" + body + struct.pack("<H", loader.crc(body))


class HostTest(unittest.TestCase):
    def test_crc_reference(self):
        self.assertEqual(loader.crc(b"123456789"), 0x29B1)

    def test_fragmented_read_and_sequence(self):
        port = Port(reply(2, 255, payload=b"\x00\xff\xa5\x5a"))
        client = loader.Loader(port)
        client.sequence = 255
        self.assertEqual(client.request(2, 0x80000013, 4), b"\x00\xff\xa5\x5a")
        self.assertEqual(client.sequence, 0)
        self.assertEqual(port.writes[2:10], bytes.fromhex("02ff130000800400"))

    def test_reject_error_response(self):
        with self.assertRaisesRegex(ValueError, "status=2"):
            loader.Loader(Port(reply(1, 0, status=2))).request(1)

    def test_reject_crc_and_identity(self):
        damaged = bytearray(reply(4, 0, payload=b"PA7\x01"))
        damaged[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            loader.Loader(Port(damaged)).request(4)
        with self.assertRaisesRegex(ValueError, "header"):
            loader.Loader(Port(reply(4, 1))).request(4)


if __name__ == "__main__":
    unittest.main()
