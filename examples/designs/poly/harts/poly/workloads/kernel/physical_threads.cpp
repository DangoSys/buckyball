#include "copy.h"
#include <array>
#include <bb_harts.h>
#include <thread.h>

#define X(hart)                                                                \
  (sizeof((const unsigned[]){0, DEVICE_HARTS_##hart##_THREAD_INDICES}) /       \
       sizeof(unsigned) -                                                      \
   1),
static constexpr unsigned counts[] = {DEVICE_HARTS};
#undef X
#define X(hart)                                                                \
  +(sizeof((const unsigned[]){0, DEVICE_HARTS_##hart##_THREAD_INDICES}) /      \
        sizeof(unsigned) -                                                     \
    1)
enum { contexts = 0 DEVICE_HARTS, words = contexts * 6 };
#undef X
alignas(8) static const unsigned char ant_image[] = {
#embed "copy.bin"
};
static unsigned done[BB_NUM_HARTS];
alignas(64) static constexpr auto expected = [] {
  std::array<uint64_t, words> data{};
  for (unsigned i = 0; i < contexts; ++i) {
    for (unsigned j = 0; j < 6; ++j)
      data[i * 6 + j] = UINT64_C(0xa5a5a5a5a5a5a5a5);
    data[i * 6 + 2] = (uint64_t(i + 1) << 32) | 0x12345678;
    data[i * 6 + 3] = (uint64_t(i + 1) << 32) | 0x87654321;
  }
  return data;
}();
alignas(64) static auto output = [] {
  auto data = expected;
  for (unsigned i = 0; i < contexts; ++i)
    data[i * 6 + 2] = data[i * 6 + 3] = UINT64_C(0xdeadbeefdeadbeef);
  return data;
}();
static const struct {
  const void *actual, *reference;
  uint64_t bytes;
} check __attribute__((section(".bbsim.check"), used, retain)) = {
    output.data(), expected.data(), sizeof(expected)};

int main() {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  unsigned slot = bb_hart_slot(hart), first = 0;
  for (unsigned i = 0; i < slot; ++i)
    first += counts[i];
  unsigned count = counts[slot];
  if (count) {
    if (thread_query(0, THREAD_FIELD_COUNT) != count)
      __builtin_trap();
    size_t bytes = sizeof(ant_image);
    struct thread_task tasks[count];
    thread_tss_write(0, ant_image, bytes);
    for (unsigned context = 0; context < count; ++context) {
      uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE);
      tasks[context] = {context + 1,
                        0,
                        bytes,
                        base,
                        base + thread_query(context, THREAD_FIELD_TLS_BYTES),
                        thread_query(context, THREAD_FIELD_SIGNATURE)};
      thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~size_t(7));
    }
    thread_acquire();
    for (unsigned context = 0; context < count; ++context) {
      unsigned at = (first + context) * 6 + 2;
      copy_task args{uintptr_t(expected.data() + at),
                     uintptr_t(output.data() + at), 1};
      thread_start_data(context, &tasks[context], &args, sizeof(args));
    }
    for (unsigned context = 0; context < count; ++context) {
      int cancelled;
      if (thread_wait(context, tasks[context].id, &cancelled) || cancelled)
        __builtin_trap();
    }
    thread_release();
  }
  __atomic_store_n(&done[slot], 1, __ATOMIC_RELEASE);
  if (hart == WORKLOAD_REPORT_HART)
    for (unsigned i = 0; i < BB_NUM_HARTS; ++i)
      while (!__atomic_load_n(&done[i], __ATOMIC_ACQUIRE)) {
      }
  return 0;
}
