#include "VScheduler.h"
#include "verilated.h"
#include <cstdio>
#include <map>
#include <stdexcept>
#include <vector>

struct Issued {
  unsigned read, write;
};
struct Reclaimed {
  unsigned previous, current;
};

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Verilated::randReset(0);
  VScheduler dut;
  std::map<unsigned, Issued> issued;
  std::vector<Reclaimed> reclaimed;
  unsigned cycle = 0;
  auto check = [](bool condition, const char *message) {
    if (!condition)
      throw std::runtime_error(message);
  };
  auto tick = [&]() {
    dut.clock = 0;
    dut.eval();
    if (dut.io_ball_issue_valid && dut.io_ball_issue_ready) {
      const unsigned slot = dut.io_ball_issue_bits_rob_id;
      issued[slot] = {dut.io_ball_issue_bits_renamedBankAccess_rd_bank_0_id,
                      dut.io_ball_issue_bits_renamedBankAccess_wr_bank_id};
      std::printf("cycle=%u issue slot=%u read=%u write=%u\n", cycle, slot,
                  issued[slot].read, issued[slot].write);
    }
    if (dut.io_mapping_valid && dut.io_mapping_ready &&
        dut.io_mapping_bits_retire) {
      reclaimed.push_back(
          {dut.io_mapping_bits_previous, dut.io_mapping_bits_vbank});
      std::printf("cycle=%u reclaim previous=%u current=%u\n", cycle,
                  dut.io_mapping_bits_previous, dut.io_mapping_bits_vbank);
    }
    dut.clock = 1;
    dut.eval();
    dut.clock = 0;
    dut.eval();
    ++cycle;
  };
  auto wait = [&](auto condition) {
    for (unsigned n = 0; n < 100; ++n) {
      dut.eval();
      if (condition())
        return;
      tick();
    }
    throw std::runtime_error("Handshake timed out");
  };
  dut.reset = 1;
  for (unsigned i = 0; i < 4; ++i)
    tick();
  dut.reset = 0;
  dut.io_ball_issue_ready = 1;
  dut.io_memory_issue_ready = 1;
  dut.io_mapping_ready = 1;
  dut.io_thread_id = 1;
  auto send = [&](bool read, bool write) {
    dut.io_decode_cmd_i_bits_domain_id = 3;
    dut.io_decode_cmd_i_bits_cmd_funct = 0x63;
    dut.io_decode_cmd_i_bits_cmd_pc = 4 + cycle * 4;
    dut.io_decode_cmd_i_bits_bankAccess_rd_bank_0_valid = read;
    dut.io_decode_cmd_i_bits_bankAccess_rd_bank_0_id = 1;
    dut.io_decode_cmd_i_bits_bankAccess_wr_bank_valid = write;
    dut.io_decode_cmd_i_bits_bankAccess_wr_bank_id = 1;
    dut.io_decode_cmd_i_valid = 1;
    wait([&] { return dut.io_decode_cmd_i_ready; });
    const unsigned slot = dut.io_allocation_bits_rob_id;
    tick();
    dut.io_decode_cmd_i_valid = 0;
    return slot;
  };
  auto complete = [&](unsigned slot) {
    dut.io_ball_complete_bits_rob_id = slot;
    dut.io_ball_complete_valid = 1;
    wait([&] { return dut.io_ball_complete_ready; });
    tick();
    dut.io_ball_complete_valid = 0;
  };
  try {
    const auto reader = send(true, false);
    const auto writer1 = send(false, true);
    const auto writer2 = send(false, true);
    wait([&] {
      return issued.count(reader) && issued.count(writer1) &&
             issued.count(writer2);
    });
    check(issued[reader].read == 1, "Old reader lost its binding");
    const auto first = issued[writer1].write, second = issued[writer2].write;
    check(first != 1 && second != first,
          "Consecutive writes must use distinct aliases");
    complete(writer2);
    complete(writer1);
    for (unsigned i = 0; i < 12; ++i)
      tick();
    check(reclaimed.empty(),
          "Out-of-order completion freed an alias before retirement");
    dut.io_mapping_ready = 0;
    complete(reader);
    wait([&] { return dut.io_mapping_valid && dut.io_mapping_bits_retire; });
    for (unsigned i = 0; i < 10; ++i) {
      check(dut.io_mapping_bits_previous == 1 &&
                dut.io_mapping_bits_vbank == first,
            "Reclaim payload changed under backpressure");
      check(!(dut.io_retired & ((1u << writer1) | (1u << writer2))),
            "ROB retired a writer before reclaim was accepted");
      tick();
    }
    dut.io_mapping_ready = 1;
    wait([&] { return reclaimed.size() == 2; });
    check(reclaimed[0].previous == 1 && reclaimed[0].current == first,
          "First stale binding mismatch");
    check(reclaimed[1].previous == first && reclaimed[1].current == second,
          "Second stale binding mismatch");
    const auto writer3 = send(true, true);
    wait([&] { return issued.count(writer3); });
    check(issued[writer3].read == second,
          "Read-modify-write did not read the previous binding");
    check(issued[writer3].write == first,
          "Retired alias was not returned to the free list");
    complete(writer3);
    wait([&] { return reclaimed.size() == 3 && dut.io_idle; });
    check(reclaimed[2].previous == second && reclaimed[2].current == first,
          "Reused alias retirement mismatch");
    std::puts("PASS: out-of-order completion, reclaim backpressure, repeated "
              "rename and alias reuse");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL at cycle %u: %s\n", cycle, error.what());
    return 1;
  }
  dut.final();
}
