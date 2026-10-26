#include <multithread.h>
extern "C" int toy_pipeline(uint32_t instance);
static volatile uint32_t completed, failed;
int main() {
  const uint32_t hart = bb_get_hart_id();
  const uint32_t expected =
      bb_profile_core_count(BB_COMPONENT_WORKLOAD_PROFILE);
  if (!expected)
    return 1;
  if (bb_hart_profile(hart) == BB_COMPONENT_WORKLOAD_PROFILE) {
    __atomic_fetch_or(&failed, toy_pipeline(bb_hart_slot(hart)) != 0,
                      __ATOMIC_RELEASE);
    __atomic_add_fetch(&completed, 1, __ATOMIC_RELEASE);
  }
  if (hart == BB_REPORT_HART) {
    while (__atomic_load_n(&completed, __ATOMIC_ACQUIRE) != expected)
      asm volatile("nop");
    return __atomic_load_n(&failed, __ATOMIC_ACQUIRE) != 0;
  }
  for (;;)
    asm volatile("wfi");
}
