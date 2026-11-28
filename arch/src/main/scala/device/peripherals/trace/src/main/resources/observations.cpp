#include <cstdint>

static unsigned count;
static unsigned channel, word, low, high;

extern "C" void dpi_trace_write(unsigned ch, unsigned reg, unsigned lo,
                                unsigned hi) {
  ++count;
  channel = ch;
  word = reg;
  low = lo;
  high = hi;
}

extern "C" unsigned trace_observed(unsigned *ch, unsigned *reg, unsigned *lo,
                                   unsigned *hi) {
  *ch = channel;
  *reg = word;
  *lo = low;
  *hi = high;
  return count;
}
