#include <isa/syntzulu.h>
#include <stdio.h>

static volatile uint32_t row[4] __attribute__((aligned(128)));
// Each output pair sums 16 input weights. The two lane sums are 40 and 104
// for pair zero; each subsequent pair scales both by (index + 1).

int main(void) {
  bb_mem_alloc(0, 1, 1);
  for (unsigned address = 0; address < 128; ++address) {
    uint64_t weights = address < 96 ? UINT64_C(0x0102040810203040)
                                    : UINT64_C(0x0102030405060708) * (1 + ((address >> 2) & 7));
    syntzulu_weight(0, 0, address, (uint32_t)weights);
    syntzulu_weight(0, 1, address, (uint32_t)(weights >> 32));
  }
  const uint16_t frames[] = {1, 0, 0x8000, 0xffff};
  uint16_t potential[16] = {0};
  for (unsigned frame = 0; frame < 4; ++frame) {
    syntzulu_run(0, frames[frame]);
    for (unsigned i = 0; i < 8; ++i) {
      // All hidden units fire for each nonempty frame with these positive weights.
      for (unsigned lane = 0; lane < 2; ++lane) {
        unsigned index = 2 * i + lane;
        unsigned voltage = potential[index] * 3672u / 4096u;
        if (frames[frame]) voltage += (lane ? 40u : 104u) * (i + 1);
        potential[index] = voltage >= 1024 ? 0 : voltage;
      }
      uint32_t expected = ((uint32_t)potential[2*i+1] << 16) | potential[2*i];
      if (syntzulu_read(0, i, row) != expected) return 1 + 8 * frame + i;
    }
  }
  bb_mem_release(0);
  puts("SYNtzulu PASS: all 32 output words match standalone RTL");
  return 0;
}
