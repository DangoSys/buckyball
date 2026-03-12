#include "monitor/halt.h"
#include "bdb.h"
#include <stdio.h>

// Called from HaltDPI.sv via DPI-C when ebreak exception is detected
void dpi_sim_halt(void) {
  printf("simulation success\n");
  sim_exit();
}
