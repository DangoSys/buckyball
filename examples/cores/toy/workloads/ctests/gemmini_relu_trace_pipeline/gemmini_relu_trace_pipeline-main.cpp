#include <multicore.h>

extern "C" int toy_core(core_id_t);
static volatile uint32_t completed;
static volatile uint32_t failed;

int main() {
  core_id_t id = bb_get_core_id();
  if (id.tile == 0 &&
      bb_topology_core_profile(id) == BB_CORE_WORKLOAD_PROFILE) {
    __atomic_fetch_or(&failed, toy_core(id) != 0, __ATOMIC_RELEASE);
    __atomic_add_fetch(&completed, 1, __ATOMIC_RELEASE);
  }
  if (id.tile == 0 && id.core == 0) {
    uint32_t expected =
        bb_topology_profile_cores_per_tile(BB_CORE_WORKLOAD_PROFILE);
    while (__atomic_load_n(&completed, __ATOMIC_ACQUIRE) != expected)
      asm volatile("nop");
    return __atomic_load_n(&failed, __ATOMIC_ACQUIRE) != 0;
  }
  for (;;)
    asm volatile("wfi");
}
