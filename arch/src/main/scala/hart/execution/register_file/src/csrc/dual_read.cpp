#include "VRegisterFile.h"
#include "verilated.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <random>
int main() {
  VerilatedContext context;
  VRegisterFile dut(&context);
  std::array<uint64_t, 32> reference{};
  for (unsigned i = 1; i < 32; ++i) {
    dut.io_write_valid = 1;
    dut.io_write_bits_address = i;
    dut.io_write_bits_data = reference[i] = 0x1234000000000000ULL + i;
    dut.clock = 0;
    dut.eval();
    dut.clock = 1;
    dut.eval();
  }
  std::mt19937_64 random(0x6447);
  for (unsigned i = 0; i < 60000; ++i) {
    unsigned a = random() % 32, b = random() % 32, w = random() % 32;
    bool valid = random() % 2;
    uint64_t data = random();
    dut.io_readAddress_0 = a;
    dut.io_readAddress_1 = b;
    dut.io_write_valid = valid;
    dut.io_write_bits_address = w;
    dut.io_write_bits_data = data;
    dut.clock = 0;
    dut.eval();
    assert(dut.io_readData_0 == (a == 0            ? 0
                                 : valid && a == w ? data
                                                   : reference[a]));
    assert(dut.io_readData_1 == (b == 0            ? 0
                                 : valid && b == w ? data
                                                   : reference[b]));
    dut.clock = 1;
    dut.eval();
    if (valid && w)
      reference[w] = data;
  }
  puts("PASS: 60000 dual-read register-file checks, write bypass and x0 "
       "suppression");
}
