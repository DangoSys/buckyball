#include <multicore.h>

extern volatile uint32_t test_done;
extern volatile int test_result;

#ifdef __cplusplus
extern "C"
#endif
    int prefill_main(core_id_t id) {
  if (id.core == 0) {
    while (!test_done)
      asm volatile("nop");
    asm volatile("fence rw, rw" ::: "memory");
    return test_result;
  }
  for (;;)
    asm volatile("wfi");
}
