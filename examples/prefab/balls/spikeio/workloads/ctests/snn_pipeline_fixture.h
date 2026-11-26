#include <isa/spikeio.h>
#include <isa/syntzulu.h>
#include <stdio.h>

enum { FEATURES, SPIKES, POTENTIALS, LABEL };
static const volatile int8_t samples[4][16] __attribute__((aligned(128))) = {
  {6,5,0,-1,-128,4,5,3,1,0,4,0,2,1,5,0},
  {0}, {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127},
  {6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21}
};
static volatile uint32_t row[4] __attribute__((aligned(128)));

static uint32_t read_row(unsigned bank) {
  bb_mvout((uintptr_t)row, bank, 1, 1);
  asm volatile("fence rw,rw" ::: "memory");
  return row[0];
}

static inline uint64_t cycles(void) {
  uint64_t value;
  asm volatile("rdcycle %0" : "=r"(value) :: "memory");
  return value;
}

static unsigned host_label(const volatile int8_t *input, int bulk) {
  uint16_t spikes = 0;
  for (unsigned i = 0; i < 16; ++i)
    if (input[i] > 5) spikes |= 1u << i;
  static volatile uint32_t packed[4] __attribute__((aligned(128)));
  static volatile uint32_t outputs[8][4] __attribute__((aligned(128)));
  if (bulk) {
    packed[0] = spikes;
    asm volatile("fence rw,rw" ::: "memory");
    bb_mvin((uintptr_t)packed, SPIKES, 1, 1);
    syntzulu_run_bank(SPIKES, POTENTIALS);
    bb_mvout((uintptr_t)outputs, POTENTIALS, 8, 1);
    asm volatile("fence rw,rw" ::: "memory");
  } else {
    syntzulu_run(LABEL, spikes);
  }
  int best = -32769;
  unsigned winner = 0;
  for (unsigned pair = 0; pair < 8; ++pair) {
    uint32_t word = bulk ? outputs[pair][0] : syntzulu_read(LABEL, pair, row);
    for (unsigned lane = 0; lane < 2; ++lane) {
      int value = (int16_t)(word >> (16 * lane));
      if (value > best) {
        best = value;
        winner = 2 * pair + lane;
      }
    }
  }
  return winner;
}

static unsigned bank_label(const volatile int8_t *input) {
  bb_mvin((uintptr_t)input, FEATURES, 1, 1);
  spike_encode(FEATURES, SPIKES, 5);
  syntzulu_run_bank(SPIKES, POTENTIALS);
  spike_argmax(POTENTIALS, LABEL);
  return read_row(LABEL);
}

// Untimed differential check of every signed voltage, beyond the winning index.
static int check_potentials(unsigned mask, uint16_t previous[16]) {
  static volatile uint32_t rows[8][4] __attribute__((aligned(128)));
  bb_mvout((uintptr_t)rows, POTENTIALS, 8, 1);
  asm volatile("fence rw,rw" ::: "memory");
  for (unsigned pair = 0; pair < 8; ++pair) {
    for (unsigned lane = 0; lane < 2; ++lane) {
      unsigned index = 2 * pair + lane;
      unsigned v = previous[index] * 3672u / 4096u;
      if (mask) v += (lane ? 40u : 104u) * (pair + 1);
      previous[index] = v >= 1024 ? 0 : v;
    }
    uint32_t word = previous[2*pair] | ((uint32_t)previous[2*pair+1] << 16);
    if (rows[pair][0] != word || rows[pair][1] || rows[pair][2] || rows[pair][3]) return 0;
  }
  return 1;
}

enum fixture_mode { HOST, BULK_HOST, PIPELINE, CHECK };
static int run_fixture(enum fixture_mode mode) {

  for (unsigned bank = FEATURES; bank <= LABEL; ++bank) bb_mem_alloc(bank, 1, 1);
  for (unsigned addr = 0; addr < 128; ++addr) {
    uint64_t weights = addr < 96 ? UINT64_C(0x0102040810203040)
        : UINT64_C(0x0102030405060708) * (1 + ((addr >> 2) & 7));
    syntzulu_weight(LABEL, 0, addr, (uint32_t)weights);
    syntzulu_weight(LABEL, 1, addr, (uint32_t)(weights >> 32));
  }
  const unsigned labels[] = {14, 14, 8, 15};
  const unsigned masks[] = {1,0,0x8000,0xffff};
  uint64_t total = 0;
  uint16_t previous[16] = {0};
  for (unsigned frame = 0; frame < 4; ++frame) {
    uint64_t start = cycles();
    unsigned label = mode >= PIPELINE ? bank_label(samples[frame]) : host_label(samples[frame], mode == BULK_HOST);
    total += cycles() - start;
    if (label != labels[frame]) return 1 + frame;
    if (mode == CHECK && !check_potentials(masks[frame], previous)) return 9 + frame;
  }
  // Check the encoder separately, outside the measurement and after all four frames.
  if (mode >= PIPELINE) {
    for (unsigned frame = 0; frame < 4; ++frame) {
      bb_mvin((uintptr_t)samples[frame], FEATURES, 1, 1);
      spike_encode(FEATURES, SPIKES, 5);
      if (read_row(SPIKES) != masks[frame]) return 5 + frame;
    }
  }
  if (mode == CHECK) {
    // The maximum is negative and occurs twice, at indices 3 and 12.
    static uint32_t tied[8][4] __attribute__((aligned(128)));
    for (unsigned i = 0; i < 8; ++i) tied[i][0] = UINT32_C(0x80008000);
    tied[1][0] = UINT32_C(0xffff8000);
    tied[6][0] = UINT32_C(0x8000ffff);
    bb_mvin((uintptr_t)tied, POTENTIALS, 8, 1);
    spike_argmax(POTENTIALS, LABEL);
    if (read_row(LABEL) != 3) return 13;
  }
  char count[] = "four-frame cycles: 0x0000000000000000";
  for (unsigned i = 0; i < 16; ++i) {
    count[sizeof(count)-2-i] = "0123456789abcdef"[total & 15];
    total >>= 4;
  }
  puts(count);
  for (unsigned bank = FEATURES; bank <= LABEL; ++bank) bb_mem_release(bank);
  puts("SNN pipeline PASS");
  return 0;
}
