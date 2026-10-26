#include "ant_corelink_reduce.h"
#include <bbhw/isa/isa.h>
#include <corelink.h>
#include <instances.h>
#include <isa/f32add.h>
#include <params.h>
#include <thread.h>
_Static_assert(
    BB_CORE_NUM == BB_THREAD_CORE_COUNT + 1 && BB_THREAD_CORE_COUNT >= 2 &&
        (BB_THREAD_CORE_COUNT & (BB_THREAD_CORE_COUNT - 1)) == 0,
    "Reduction needs main and a power-of-two thread-bearing core count");
static float input[BB_THREAD_CORE_COUNT][272] __attribute__((aligned(64)));
static float output[BB_THREAD_CORE_COUNT][16] __attribute__((aligned(64)));
static unsigned ready[BB_THREAD_CORE_COUNT], broadcast,
    done[BB_THREAD_CORE_COUNT];
enum { WINDOW = 4 };
static uint64_t shared_base[BB_THREAD_CORE_COUNT], main_base[WINDOW];
static unsigned main_ready, batch;
static float partial(unsigned core, unsigned i) {
  return (float)(i + 1) *
         (core % 2 ? -(float)(core + 1) * 0.5f : (float)(core + 1) * 1.25f);
}
static void controller(unsigned core) {
  ant_corelink_reduce_prepare(core, input[core], output[core]);
  struct ant_corelink_reduce_args args = {0, (uintptr_t)input[core],
                                          (uintptr_t)output[core]};
  ant_corelink_reduce_phase(core, &args);
  shared_base[core] = corelink_shared_export(0, BB_SHARED_BANK_BASE, 0);
  while (!__atomic_load_n(&main_ready, __ATOMIC_ACQUIRE) ||
         __atomic_load_n(&batch, __ATOMIC_ACQUIRE) != core / WINDOW) {
  }
  corelink_transfer(shared_base[core], 0, main_base[core % WINDOW], 16);
  __atomic_store_n(&ready[core], 1, __ATOMIC_RELEASE);
  while (!__atomic_load_n(&broadcast, __ATOMIC_ACQUIRE)) {
  }
  args.phase = 1;
  ant_corelink_reduce_phase(core, &args);
  for (unsigned i = 0; i < 4; ++i) {
    float expected = 0.0f;
    for (unsigned rank = 0; rank < BB_THREAD_CORE_COUNT; ++rank)
      expected += partial(rank, i);
    if (output[core][i] != partial(core, i) || output[core][i + 4] != expected)
      __builtin_trap();
  }
  corelink_shared_release(0, BB_SHARED_BANK_BASE, 0);
  args.phase = 2;
  ant_corelink_reduce_phase(core, &args);
  __atomic_store_n(&done[core], 1, __ATOMIC_RELEASE);
}
static void reduce(void) {
  unsigned groups =
      BB_THREAD_CORE_COUNT < WINDOW ? BB_THREAD_CORE_COUNT : WINDOW;
  unsigned source = 32, accumulator = 33, target = 34;
  bb_mem_alloc(source, 1, groups);
  bb_mem_alloc(accumulator, 1, 1);
  bb_mem_alloc(target, 1, 1);
  for (unsigned group = 0; group < groups; ++group)
    main_base[group] = corelink_shared_export(0, source, group);
  __atomic_store_n(&main_ready, 1, __ATOMIC_RELEASE);
  for (unsigned core = 0; core < BB_THREAD_CORE_COUNT; ++core) {
    while (!__atomic_load_n(&ready[core], __ATOMIC_ACQUIRE)) {
    }
    bb_f32add(source, accumulator, target, 1, core % WINDOW, core == 0);
    unsigned previous = accumulator;
    accumulator = target;
    target = previous;
    if (core % WINDOW == WINDOW - 1) {
      __atomic_store_n(&batch, core / WINDOW + 1, __ATOMIC_RELEASE);
    }
  }
  uint64_t result = corelink_shared_export(0, accumulator, 0);
  for (unsigned core = 0; core < BB_THREAD_CORE_COUNT; ++core)
    corelink_transfer(result, core + 1, shared_base[core] + 16, 16);
  __atomic_store_n(&broadcast, 1, __ATOMIC_RELEASE);
  for (unsigned core = 0; core < BB_THREAD_CORE_COUNT; ++core)
    while (!__atomic_load_n(&done[core], __ATOMIC_ACQUIRE)) {
    }
  corelink_shared_release(0, accumulator, 0);
  for (unsigned group = 0; group < groups; ++group)
    corelink_shared_release(0, source, group);
  bb_mem_release(source);
  bb_mem_release(accumulator);
  bb_mem_release(target);
}
int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart > BB_THREAD_CORE_COUNT || corelink_query(CORELINK_CORE_ID) != hart ||
      corelink_query(CORELINK_CORE_COUNT) != BB_THREAD_CORE_COUNT + 1 ||
      corelink_query(CORELINK_BANK_BYTES) < 48 ||
      corelink_query(CORELINK_SHARED_BYTES) <
          corelink_query(CORELINK_BANK_BYTES) + BB_THREAD_CORE_COUNT * 16)
    __builtin_trap();
  if (hart)
    controller(hart - 1);
  else
    reduce();
  return 0;
}
