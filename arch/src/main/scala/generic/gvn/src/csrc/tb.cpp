// Drives CheckerOnChip from a stimulus file written by model.py and prints one
// line per verdict. Stimulus lines (one cycle each unless noted):
//   c <addr> <data hex>                                  config word write
//   b <aw valid> <addr hex> <len> <w valid> <data hex> <strb hex>  snooped AW/W
//   f                                                    fence; waits for the
//   verdict s <text>                                             start of a
//   case (echoed)
#include "VCheckerOnChip.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

static void set_wide(VlWide<4> &dst, const std::string &hex) {
  std::string h = std::string(32 - hex.size(), '0') + hex;
  for (int w = 0; w < 4; ++w)
    dst[w] = std::stoul(h.substr(32 - 8 * (w + 1), 8), nullptr, 16);
}

template <typename T> static std::string wide_hex(const T &value, int words) {
  std::string out;
  char buf[9];
  for (int w = words - 1; w >= 0; --w) {
    snprintf(buf, sizeof(buf), "%08x", value[w]);
    out += buf;
  }
  return out;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  if (argc != 2)
    throw std::runtime_error("usage: tb <stimulus>");
  std::ifstream in(argv[1]);
  if (!in)
    throw std::runtime_error("cannot open stimulus");
  VCheckerOnChip dut;
  uint64_t cycle = 0;
  uint64_t last_beat = 0;
  auto tick = [&] {
    dut.clock = 0;
    dut.eval();
    dut.clock = 1;
    dut.eval();
    ++cycle;
  };
  auto clear = [&] {
    dut.io_aw_valid = 0;
    dut.io_w_valid = 0;
    dut.io_config_valid = 0;
    dut.io_fence = 0;
  };
  clear();
  dut.reset = 1;
  for (int i = 0; i < 4; ++i)
    tick();
  dut.reset = 0;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string op;
    s >> op;
    clear();
    if (op == "c") {
      std::string addr, data;
      s >> addr >> data;
      dut.io_config_valid = 1;
      dut.io_config_bits_addr = std::stoul(addr, nullptr, 16);
      dut.io_config_bits_data = std::stoull(data, nullptr, 16);
      tick();
    } else if (op == "b") {
      int awv, wv, len;
      std::string addr, data, strb;
      s >> awv >> addr >> len >> wv >> data >> strb;
      dut.io_aw_valid = awv;
      dut.io_aw_bits_addr = std::stoull(addr, nullptr, 16);
      dut.io_aw_bits_len = len;
      dut.io_aw_bits_size = 4;
      dut.io_aw_bits_burst = 1;
      dut.io_w_valid = wv;
      set_wide(dut.io_w_bits_data, data);
      dut.io_w_bits_strb = std::stoul(strb, nullptr, 16);
      dut.io_w_bits_last = 0;
      tick();
      if (wv)
        last_beat = cycle;
    } else if (op == "f") {
      while (!dut.io_fenceReady)
        tick();
      dut.io_fence = 1;
      uint64_t fence = cycle;
      tick();
      dut.io_fence = 0;
      while (!dut.io_verdict_valid) {
        tick();
        if (cycle - fence > 100000)
          throw std::runtime_error("verdict timeout");
      }
      printf("v %u %u %u %u %08x %d %d %llx %s %llu %llu %u\n",
             dut.io_verdict_bits_entry, dut.io_verdict_bits_hashFail,
             dut.io_verdict_bits_rangeFail, dut.io_verdict_bits_kurtosisFail,
             dut.io_verdict_bits_hash, (int16_t)dut.io_verdict_bits_max,
             (int16_t)dut.io_verdict_bits_min,
             (unsigned long long)dut.io_verdict_bits_s2,
             wide_hex(dut.io_verdict_bits_s4, 3).c_str(),
             (unsigned long long)(cycle - fence),
             (unsigned long long)(cycle - last_beat), dut.io_fault);
    } else if (op == "s") {
      printf("%s\n", line.c_str());
    } else {
      throw std::runtime_error("bad stimulus line: " + line);
    }
  }
  return 0;
}
