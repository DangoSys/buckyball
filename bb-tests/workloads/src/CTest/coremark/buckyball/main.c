#include <bbsw/test.h>
#include <stdio.h>

int coremark_main(void);

int main(void) {
#ifdef MULTICORE
  unsigned long hartid;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid != 0) {
    while (1) {
      __asm__ volatile("wfi");
    }
  }
#endif

  int ret = coremark_main();
  if (ret == 0) {
    bb_test_pass();
    while (1) {
    }
  }

  printf("CoreMark failed with code %d\n", ret);
  bb_test_fail();
}
