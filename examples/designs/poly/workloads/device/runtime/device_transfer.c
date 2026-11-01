#include <bb_cores.h>
#include <corelink.h>
#include <stdint.h>

static unsigned ready, received[BB_NUM_CORES], returned[BB_NUM_CORES];
static const uint64_t guard = UINT64_C(0xa5a5a5a5a5a5a5a5);
static const uint64_t change = UINT64_C(0x5a5a5a5a5a5a5a5a);

int main(void) {
  uintptr_t core;
  __asm__ volatile("csrr %0, mhartid" : "=r"(core));
  if (corelink_query(CORELINK_CORE_ID) != core ||
      corelink_query(CORELINK_CORE_COUNT) != BB_NUM_CORES ||
      corelink_query(CORELINK_CAPACITY) < BB_NUM_CORES * 4096 + 96)
    return 1;
  if (core == 0) {
    for (unsigned peer = 1; peer < BB_NUM_CORES; ++peer) {
      if (!corelink_connected(peer))
        return 2;
      for (unsigned word = 0; word < 14; ++word)
        corelink_write64(peer * 4096 - 32 + word * 8, guard);
    }
    __atomic_store_n(&ready, 1, __ATOMIC_RELEASE);
    for (unsigned peer = 1; peer < BB_NUM_CORES; ++peer) {
      while (!__atomic_load_n(&received[peer], __ATOMIC_ACQUIRE)) {
      }
      const uint64_t base = peer * 4096 - 16;
      for (unsigned word = 0; word < 12; ++word) {
        unsigned bytes = word * 8 < 69 ? 69 - word * 8 : 0;
        uint64_t mask =
            bytes >= 8 ? UINT64_MAX : (UINT64_C(1) << (bytes * 8)) - 1;
        uint64_t value = ((uint64_t)peer << 32) | (word * 17 + 3);
        if (corelink_read64(base + word * 8) !=
            ((value & mask) | (guard & ~mask)))
          return 3;
        if (bytes)
          corelink_write64(base + word * 8, value ^ change);
      }
      if (corelink_read64(base - 8) != guard)
        return 4;
      corelink_transfer(base, peer, 8192 - 16, 69);
      __atomic_store_n(&returned[peer], 1, __ATOMIC_RELEASE);
    }
  } else {
    if (!corelink_connected(0))
      return 5;
    // Two poly cores have no route between them.
    for (unsigned peer = 1; peer < BB_NUM_CORES; ++peer)
      if (corelink_connected(peer))
        return 8;
    for (unsigned word = 0; word < 12; ++word) {
      uint64_t value = ((uint64_t)core << 32) | (word * 17 + 3);
      corelink_write64(4096 - 16 + word * 8, value);
      corelink_write64(8192 - 16 + word * 8, guard);
    }
    corelink_write64(8192 - 24, guard);
    while (!__atomic_load_n(&ready, __ATOMIC_ACQUIRE)) {
    }
    corelink_transfer(4096 - 16, 0, core * 4096 - 16, 69);
    __atomic_store_n(&received[core], 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&returned[core], __ATOMIC_ACQUIRE)) {
    }
    for (unsigned word = 0; word < 12; ++word) {
      unsigned bytes = word * 8 < 69 ? 69 - word * 8 : 0;
      uint64_t mask =
          bytes >= 8 ? UINT64_MAX : (UINT64_C(1) << (bytes * 8)) - 1;
      uint64_t value = (((uint64_t)core << 32) | (word * 17 + 3)) ^ change;
      if (corelink_read64(8192 - 16 + word * 8) !=
          ((value & mask) | (guard & ~mask)))
        return 6;
    }
    if (corelink_read64(8192 - 24) != guard)
      return 7;
  }
  return 0;
}
