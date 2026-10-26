#include "buckyball.h"
#include <isa/bdb_counter.h>
#include <stdio.h>

int main(void) {
  for (int counter = 0; counter < 4; ++counter)
    bdb_counter_start(counter, 0xc000 + counter);
  for (int counter = 3; counter >= 0; --counter)
    bdb_counter_stop(counter);
  printf("bdb counter multi-id PASS\n");
  return 0;
}
