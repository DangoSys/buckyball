#include "buckyball.h"
#include <isa/bdb_counter.h>
#include <stdio.h>

int main(void) {
  bdb_counter_start(0, 0xa001);
  volatile int value = 0;
  for (int i = 0; i < 16; ++i)
    value += i;
  bdb_counter_stop(0);
  printf("bdb counter start-stop PASS %d\n", value);
  return 0;
}
