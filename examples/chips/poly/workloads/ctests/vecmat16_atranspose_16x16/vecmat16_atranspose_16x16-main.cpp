#include <multicore.h>

extern "C" int prefill_main(core_id_t);
extern "C" int decode_main(core_id_t);
extern "C" {
volatile uint32_t test_done = 0;
volatile int test_result = 1;
}

int main() {
  core_id_t id = bb_get_core_id();
  if (id.core < 3)
    return prefill_main(id);
  if (id.core < 5)
    return decode_main(id);
  return 1;
}
