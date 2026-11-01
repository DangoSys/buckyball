#include "event.h"
#include "soc.h"

static void emit(const struct trace_event *event) {
  volatile uint64_t *port =
      (volatile uint64_t *)(TRACE_BASE + CORE_HART_ID * TRACE_STRIDE);
  const unsigned long long *words = (const unsigned long long *)event;
  for (unsigned i = 0; i < TRACE_WORDS; ++i) {
    port[i] = words[i];
    asm volatile("fence iorw, iorw" ::: "memory");
  }
}

int main(void) {
  struct trace_event outer = {TRACE_BEGIN,
                              7,
                              CORE_HART_ID,
                              0,
                              0,
                              CORE_HART_ID,
                              1,
                              {7, -1ULL, -1ULL, -1ULL},
                              100};
  struct trace_event inner = {TRACE_BEGIN,  8, CORE_HART_ID,         0,  0,
                              CORE_HART_ID, 2, {7, 8, -1ULL, -1ULL}, 110};
  emit(&outer);
  emit(&inner);
  inner.event = TRACE_END;
  inner.cycle = 130;
  emit(&inner);
  outer.event = TRACE_END;
  outer.cycle = 150;
  emit(&outer);
  return 0;
}
