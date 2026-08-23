#include <bbsw/test.h>
#include <stdio.h>

int coremark_main(void);

int main(void) {
  int ret = coremark_main();
  if (ret == 0) {
    bb_test_pass();
    while (1) {
    }
  }

  printf("CoreMark failed with code %d\n", ret);
  bb_test_fail();
}
