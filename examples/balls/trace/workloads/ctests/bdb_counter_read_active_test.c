#include "buckyball.h"
#include <isa/bdb_counter.h>
#include <stdio.h>

int main(void) {
  bdb_counter_start(1, 0xa002);
  volatile int value = 0;
  for (int i = 0; i < 8; ++i)
    value += i;
  bdb_counter_read(1);
  bdb_counter_stop(1);
  printf("bdb counter active-read PASS %d\n", value);
  return 0;
}
