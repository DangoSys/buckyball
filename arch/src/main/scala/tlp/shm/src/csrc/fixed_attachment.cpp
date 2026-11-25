#include "VMeshSharedMem.h"
#include "verilated.h"
#include <cstdio>
#include <stdexcept>

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::randReset(0);
  VMeshSharedMem dut;
  unsigned cycle = 0;
  auto check = [](bool condition, const char *message) {
    if (!condition)
      throw std::runtime_error(message);
  };
  auto tick = [&]() {
    dut.clock = 0;
    dut.eval();
    dut.clock = 1;
    dut.eval();
    dut.clock = 0;
    dut.eval();
    ++cycle;
  };
  auto wait = [&](auto condition) {
    for (unsigned n = 0; n < 300; ++n) {
      dut.eval();
      if (condition())
        return;
      tick();
    }
    throw std::runtime_error("Mesh response timed out");
  };
  dut.io_bankWrites_0_ready = 1;
  dut.io_bankWrites_1_ready = 1;
  dut.io_bankWrites_2_ready = 1;
  dut.io_bankWrites_3_ready = 1;
  dut.io_bankWrites_4_ready = 1;
  dut.io_bankWrites_5_ready = 1;
  dut.io_bankWrites_6_ready = 1;
  dut.io_bankWrites_7_ready = 1;
  dut.io_bankWrites_8_ready = 1;
  dut.io_bankWrites_9_ready = 1;
  dut.io_bankWrites_10_ready = 1;
  dut.io_bankWrites_11_ready = 1;
  dut.reset = 1;
  for (unsigned i = 0; i < 4; ++i)
    tick();
  dut.reset = 0;
  try {
    // Prototype thread 2 has two ports, both fixed at bank node 4.
    // Write through its second port to remote bank 11, then read through its
    // first.
    dut.io_channels_3_request_bits_tdest = 11;
    dut.io_channels_3_request_bits_tid = 9;
    dut.io_channels_3_request_bits_tuser = (8u << 1) | 1u;
    dut.io_channels_3_request_bits_tkeep = 0xffff;
    dut.io_channels_3_request_bits_tlast = 1;
    for (unsigned word = 0; word < 4; ++word)
      dut.io_channels_3_request_bits_tdata[word] = 0x12340000u + word;
    dut.io_channels_3_request_valid = 1;
    wait([&] { return dut.io_channels_3_request_ready; });
    std::printf("cycle=%u write remote bank=11 row=8 through thread=2 port=1\n",
                cycle);
    tick();
    dut.io_channels_3_request_valid = 0;
    wait([&] { return dut.io_channels_3_response_valid; });
    check(dut.io_channels_3_response_bits_tid == 9 &&
              dut.io_channels_3_response_bits_tuser == 2,
          "Write response returned to the wrong endpoint");
    for (unsigned i = 0; i < 5; ++i) {
      check(dut.io_channels_3_response_valid &&
                dut.io_channels_3_response_bits_tid == 9,
            "Write response changed under backpressure");
      tick();
    }
    dut.io_channels_3_response_ready = 1;
    tick();
    dut.io_channels_2_request_bits_tdest = 11;
    dut.io_channels_2_request_bits_tid = 10;
    dut.io_channels_2_request_bits_tuser = 8u << 1;
    dut.io_channels_2_request_bits_tkeep = 0xffff;
    dut.io_channels_2_request_bits_tlast = 1;
    dut.io_channels_2_request_valid = 1;
    wait([&] { return dut.io_channels_2_request_ready; });
    std::printf("cycle=%u read remote bank=11 row=8 through thread=2 port=0\n",
                cycle);
    tick();
    dut.io_channels_2_request_valid = 0;
    wait([&] { return dut.io_channels_2_response_valid; });
    check(dut.io_channels_2_response_bits_tid == 10 &&
              dut.io_channels_2_response_bits_tuser == 0,
          "Read response returned to the wrong endpoint");
    for (unsigned word = 0; word < 4; ++word) {
      check(dut.io_channels_2_response_bits_tdata[word] == 0x12340000u + word,
            "Remote bank read data mismatch");
    }
    dut.io_channels_2_response_ready = 1;
    tick();
    std::puts("PASS: two ports at one fixed mesh node access a remote bank and "
              "preserve response ownership");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL at cycle %u: %s\n", cycle, error.what());
    return 1;
  }
  dut.final();
}
