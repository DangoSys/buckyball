#include "Vpebble_a7_platform.h"
#include "verilated.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int BIT = 16;
static Vpebble_a7_platform dut;
static uint64_t cycle = 0;
static int remaining = 0, bit = -1;
static unsigned byte_value = 0;
static bool previous = true;
static std::vector<unsigned char> received;
static void check(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}
static void tick(unsigned n = 1) {
  while (n--) {
    dut.clk = 0;
    dut.eval();
    Verilated::timeInc(10);
    dut.clk = 1;
    dut.eval();
    Verilated::timeInc(10);
    ++cycle;
    bool tx = dut.uart_tx_out;
    if (remaining) {
      if (--remaining == 0) {
        if (bit < 8) {
          byte_value |= (unsigned)tx << bit;
          ++bit;
          remaining = BIT;
        } else {
          check(tx, "UART stop bit");
          received.push_back(byte_value);
          bit = -1;
        }
      }
    } else if (previous && !tx) {
      bit = 0;
      byte_value = 0;
      remaining = BIT + BIT / 2;
    }
    previous = tx;
    check(!Verilated::gotFinish(), "RTL assertion/finish");
  }
}
static void send(unsigned value) {
  dut.uart_rx_in = 0;
  tick(BIT);
  for (int i = 0; i < 8; ++i) {
    dut.uart_rx_in = (value >> i) & 1;
    tick(BIT);
  }
  dut.uart_rx_in = 1;
  tick(BIT * 2);
}
static uint16_t crc(const std::vector<unsigned char> &data) {
  uint16_t c = 0xffff;
  for (unsigned b : data) {
    c ^= b << 8;
    for (int i = 0; i < 8; ++i)
      c = (c & 0x8000) ? (c << 1) ^ 0x1021 : c << 1;
  }
  return c;
}
static std::vector<unsigned char>
packet(unsigned op, unsigned seq, uint32_t addr, unsigned len,
       const std::vector<unsigned char> &data = {}, bool corrupt = false,
       unsigned status = 0) {
  std::vector<unsigned char> body = {(unsigned char)op, (unsigned char)seq};
  for (int i = 0; i < 4; ++i)
    body.push_back(addr >> (8 * i));
  body.push_back(len);
  body.push_back(len >> 8);
  body.insert(body.end(), data.begin(), data.end());
  uint16_t c = crc(body) ^ (corrupt ? 1 : 0);
  received.clear();
  send(0xa5);
  send(0x5a);
  for (auto b : body)
    send(b);
  send(c & 255);
  send(c >> 8);
  uint64_t deadline = cycle + 1000000;
  while (received.size() < 7 && cycle < deadline)
    tick();
  check(received.size() >= 7, "Loader header timeout");
  unsigned count = received[5] | received[6] << 8;
  while (received.size() < 9 + count && cycle < deadline)
    tick();
  check(received.size() == 9 + count, "Loader response length");
  check(received[0] == 0x5a && received[1] == 0xa5 &&
            received[2] == (op | 128) && received[3] == seq,
        "Loader response identity");
  check(received[4] == status,
        "Loader response status " + std::to_string(received[4]));
  std::vector<unsigned char> response(received.begin() + 2, received.end() - 2);
  check(crc(response) == (received[received.size() - 2] | received.back() << 8),
        "Loader response CRC");
  return {received.begin() + 7, received.end() - 2};
}
static std::string console_until(const std::string &ending,
                                 unsigned limit = 4000000) {
  std::string result;
  size_t seen = 0;
  while (limit--) {
    tick();
    while (seen < received.size()) {
      char c = received[seen++];
      result += c;
      std::cout << c << std::flush;
    }
    check(result.find("FAIL ") == std::string::npos &&
              result.find("TRAP ") == std::string::npos,
          "Firmware failure");
    if (result.find(ending) != std::string::npos)
      return result;
  }
  throw std::runtime_error("CPU/console timeout after " +
                           std::to_string(cycle) + " clocks");
}
int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  try {
    // Verify the synthesized BRAM image before the loader can overwrite it.
    dut.reset = 1;
    dut.boot_demo = 1;
    dut.uart_rx_in = 1;
    tick(40);
    dut.reset = 0;
    auto cold = console_until("other bytes echo\r\n");
    check(cold.find("PASS PEBBLE_A7_FULL_CHIP") != std::string::npos &&
              dut.exited && dut.exit_code == 0,
          "Cold resident image boot");
    dut.reset = 1;
    dut.boot_demo = 0;
    tick(40);
    dut.reset = 0;
    received.clear();
    tick(40);
    check(!dut.cpu_run, "CPU released before load");
    auto info = packet(4, 0, 0, 0);
    check(info.size() == 12 && info[0] == 'P' && info[2] == '7', "INFO");
    packet(3, 1, 0x80000000, 0, {}, false, 5);
    check(!dut.cpu_run, "Empty RUN accepted");
    std::ifstream file("firmware/demo.bin", std::ios::binary);
    check(file.good(), "Missing firmware");
    std::vector<unsigned char> program((std::istreambuf_iterator<char>(file)),
                                       {});
    unsigned seq = 2;
    packet(1, seq++, 0x80000000, 4, {0xde, 0xad, 0xbe, 0xef}, true, 1);
    for (size_t offset = 0; offset < program.size(); offset += 256) {
      size_t count = std::min<size_t>(256, program.size() - offset);
      std::vector<unsigned char> block(program.begin() + offset,
                                       program.begin() + offset + count);
      packet(1, seq++ & 255, 0x80000000 + offset, count, block);
      check(packet(2, seq++ & 255, 0x80000000 + offset, count) == block,
            "Firmware readback mismatch");
    }
    packet(3, seq++ & 255, 0x80000000, 0, {}, true, 1);
    check(!dut.cpu_run, "Corrupt RUN accepted");
    packet(3, seq++ & 255, 0x80000000, 0);
    received.clear();
    auto first = console_until("other bytes echo\r\n");
    check(first.find("PASS PEBBLE_A7_FULL_CHIP") != std::string::npos &&
              dut.exited && dut.exit_code == 0,
          "No firmware PASS");
    received.clear();
    send('Q');
    console_until("Q");
    received.clear();
    send('r');
    auto second = console_until("PASS PEBBLE_A7_FULL_CHIP\r\n");
    check(second.find("0000000000000002") != std::string::npos, "Rerun seed");
    // Exercise the resident BRAM image boot path and reset of CPU/accelerator.
    dut.reset = 1;
    dut.boot_demo = 1;
    tick(40);
    dut.reset = 0;
    received.clear();
    auto third = console_until("other bytes echo\r\n");
    check(third.find("PASS PEBBLE_A7_FULL_CHIP") != std::string::npos,
          "Resident boot");
    std::cout
        << "PASS: full chip cold boot, serial upload/readback, CPU, DMA, three "
           "operators, echo, rerun and reset; cycles="
        << cycle << "\n";
    dut.final();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    dut.final();
    return 1;
  }
}
