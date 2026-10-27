#include "buckyball.h"
#include <isa/bdb_counter.h>
#include <stdio.h>

int main(void) {
  bdb_counter_start(0, 0xb001);
  bdb_counter_start(1, 0xb002);
  volatile int value = 0;
  for (int i = 0; i < 8; ++i)
    value += i;
  bdb_counter_stop(1);
  bdb_counter_stop(0);
  printf("bdb counter nested-pair PASS %d\n", value);
  return 0;
}
