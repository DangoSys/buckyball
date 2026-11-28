#include <array>

struct Event {
  unsigned hart, value;
};
static std::array<Event, 3> last{};
static std::array<unsigned, 3> count{};
static std::array<int, 2> rx{-1, -1};

extern "C" void scu_uart_write(unsigned hart, unsigned ch) {
  last[0] = {hart, ch};
  ++count[0];
}

extern "C" void scu_sim_exit(unsigned hart, unsigned code) {
  last[1] = {hart, code};
  ++count[1];
}

extern "C" void scu_uart_rx_sample(unsigned hart, unsigned pop, unsigned *valid,
                                   unsigned *data) {
  last[2] = {hart, pop};
  ++count[2];
  int &byte = rx.at(hart);
  *valid = byte >= 0;
  *data = *valid ? static_cast<unsigned>(byte) : 0;
  if (pop)
    byte = -1;
}

extern "C" void scu_rx_input(unsigned hart, unsigned ch) { rx.at(hart) = ch; }

extern "C" unsigned scu_observed(unsigned kind, unsigned *hart,
                                 unsigned *value) {
  *hart = last.at(kind).hart;
  *value = last.at(kind).value;
  return count.at(kind);
}
