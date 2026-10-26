#include <instances.h>
#include <stdint.h>

static uint64_t payload[BB_CORE_NUM][8] __attribute__((aligned(64)));
static unsigned arrived[2] __attribute__((aligned(64)));
static unsigned consumed[2] __attribute__((aligned(64)));
static unsigned failed;

int main(void) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  unsigned slot = bb_hart_slot(hart);
  for (unsigned round = 0; round < 2; ++round) {
    uint64_t value = 0x5a000000ULL + round * 0x10000 + slot * 9;
    payload[slot][0] = value;
    payload[slot][7] = ~value;
    __atomic_fetch_add(&arrived[round], 1, __ATOMIC_ACQ_REL);
    while (__atomic_load_n(&arrived[round], __ATOMIC_ACQUIRE) != BB_CORE_NUM) {
    }
    for (unsigned peer = 0; peer < BB_CORE_NUM; ++peer) {
      uint64_t expected = 0x5a000000ULL + round * 0x10000 + peer * 9;
      if (payload[peer][0] != expected || payload[peer][7] != ~expected)
        __atomic_fetch_or(&failed, 1, __ATOMIC_RELAXED);
    }
    __atomic_fetch_add(&consumed[round], 1, __ATOMIC_ACQ_REL);
    while (__atomic_load_n(&consumed[round], __ATOMIC_ACQUIRE) != BB_CORE_NUM) {
    }
  }
  return __atomic_load_n(&failed, __ATOMIC_ACQUIRE);
}
