#include "bdb.h"
#include "utils/debug.h"
#include "utils/macro.h"

vluint64_t sim_time = 0;
VerilatedContext *contextp = NULL;
VerilatedFstC *tfp = NULL;

#ifdef COSIM
static VToyBuckyball *top;
#else
static VTestHarness *top;
#endif

int bb_step = 1;

void step_and_dump_wave() {
  top->eval();
  contextp->timeInc(1);
  tfp->dump(contextp->time());
  sim_time++;
}

void sim_init(int argc, char **argv) {
  contextp = new VerilatedContext;
  contextp->commandArgs(argc, argv);
  tfp = new VerilatedFstC;

#ifdef COSIM
  top = new VToyBuckyball{contextp};
#else
  top = new VTestHarness{contextp};
#endif

  contextp->traceEverOn(true);
  top->trace(tfp, 0);

  tfp->open(fst_path);
  Log("The waveform will be saved to the FST file: %s", fst_path);

  top->reset = 1;
  top->clock = 0;
  step_and_dump_wave();
  top->reset = 1;
  top->clock = 1;
  step_and_dump_wave();
  top->reset = 0;
  top->clock = 0;
  step_and_dump_wave();
}

void sim_exit() {
  contextp->timeInc(1);
  tfp->dump(contextp->time());
  tfp->close();
  printf("The wave data has been saved to the FST file: %s\n", fst_path);
  exit(0);
}

void ball_exec_once() {
  top->clock ^= 1;
  step_and_dump_wave();
  top->clock ^= 1;
  step_and_dump_wave();
#ifndef COSIM
  if (top->io_success == 1) {
    printf("simulation success\n");
    sim_exit();
  }
#endif
}

//================ main =====================//
int main(int argc, char *argv[]) {
  init_monitor(argc, argv);
  sim_init(argc, argv);
  bdb_mainloop();
  sim_exit();
}
