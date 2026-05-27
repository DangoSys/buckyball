#include <bbsw/test.h>
#include <stdint.h>

#define MMIO_SIM_EXIT ((volatile uint32_t *)0x60000000UL)

void bb_test_pass(void) { *MMIO_SIM_EXIT = 0; }

void bb_test_fail(void) {
  *MMIO_SIM_EXIT = 1;
  while (1) {
  }
}
