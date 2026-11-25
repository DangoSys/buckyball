#include "VBankHashMonitor.h"
#include "verilated.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  if (argc != 2)
    throw std::runtime_error("Expected bank row count");
  const unsigned rows = std::stoul(argv[1]);
  VBankHashMonitor dut;
  std::vector<std::array<uint32_t, 4>> contents(rows);
  auto tick = [&] {
    dut.clock = 0;
    dut.eval();
    dut.clock = 1;
    dut.eval();
    dut.clock = 0;
    dut.eval();
  };
  auto check = [&](bool value, const char *message) {
    if (!value)
      throw std::runtime_error(message);
  };
  auto rotate = [](uint32_t value, unsigned n) {
    return (value << n) | (value >> (32 - n));
  };
  auto expected = [&] {
    uint32_t hash = 0;
    for (unsigned row = 0; row < rows; ++row) {
      const auto &v = contents[row];
      if (v[0] | v[1] | v[2] | v[3])
        hash += v[0] ^ rotate(v[1], 7) ^ rotate(v[2], 13) ^ rotate(v[3], 21) ^
                rotate(row, 11);
    }
    return hash;
  };
  auto mask = [&](uint16_t bits) {
#define BYTE(i) dut.io_write_bits_mask_##i = (bits >> i) & 1
    BYTE(0);
    BYTE(1);
    BYTE(2);
    BYTE(3);
    BYTE(4);
    BYTE(5);
    BYTE(6);
    BYTE(7);
    BYTE(8);
    BYTE(9);
    BYTE(10);
    BYTE(11);
    BYTE(12);
    BYTE(13);
    BYTE(14);
    BYTE(15);
#undef BYTE
  };
  dut.reset = 1;
  dut.io_bind = 0;
  dut.io_write_valid = 0;
  tick();
  tick();
  dut.reset = 0;
  tick();
  uint32_t random = 0x12345678;
  for (unsigned round = 0; round < 3; ++round) {
    for (unsigned row = 0; row < rows; ++row) {
      check(dut.io_write_ready, "Write did not become ready");
      random = random * 1664525 + 1013904223;
      const uint16_t bytes = round == 0 ? 0xffff : random;
      dut.io_write_bits_addr = row;
      for (unsigned word = 0; word < 4; ++word) {
        random = random * 1664525 + 1013904223;
        dut.io_write_bits_data[word] = random;
        for (unsigned byte = 0; byte < 4; ++byte) {
          uint32_t m = 0xffu << (byte * 8);
          if ((bytes >> (word * 4 + byte)) & 1)
            contents[row][word] = (contents[row][word] & ~m) | (random & m);
        }
      }
      mask(bytes);
      dut.io_write_valid = 1;
      tick();
      check(!dut.io_write_ready, "Pending write did not apply backpressure");
      dut.io_write_valid = 0;
      tick();
      check(dut.io_statusHash == expected(), "Masked write hash mismatch");
    }
    if (round == 0)
      continue;
    dut.io_write_valid = 1;
    tick();
    dut.io_bind = 1;
    dut.eval();
    check(!dut.io_write_ready, "Bind accepted a concurrent write");
    tick();
    dut.io_write_valid = 0;
    dut.io_bind = 0;
    contents.assign(rows, {});
    tick();
    check(dut.io_statusHash == 0, "Bind did not cancel the pending write");
  }
  std::puts("PASS: all rows, masked overwrite, rebind, cancellation and "
            "backpressure");
  dut.final();
}
