#include "bdb.h"
#include "ioe/mmio.h"
#include "utils/debug.h"
#include "utils/macro.h"

#include <csignal>
#include <cstdlib>

vluint64_t sim_time = 0;
VerilatedContext *contextp = NULL;
VerilatedFstC *tfp = NULL;

VBBSimHarness *top;

int bb_step = 1;

#if VM_COVERAGE
static void coverage_atexit() {
  if (contextp) {
    contextp->coveragep()->write();
  }
}

static void coverage_signal_handler(int sig) {
  coverage_atexit();
  _exit(128 + sig);
}
#endif

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

  top = new VBBSimHarness{contextp};

  contextp->traceEverOn(true);
  top->trace(tfp, 0);

  tfp->open(fst_path);
  if (bdb_sim_meta_path() == nullptr) {
    Log("The waveform will be saved to the FST file: %s", fst_path);
  }

  top->reset = 1;
  top->clock = 0;
  step_and_dump_wave();
  top->reset = 1;
  top->clock = 1;
  step_and_dump_wave();
  top->reset = 0;
  top->clock = 0;
  step_and_dump_wave();

#if VM_COVERAGE
  atexit(coverage_atexit);
  signal(SIGTERM, coverage_signal_handler);
  signal(SIGINT, coverage_signal_handler);
#endif
}

void sim_exit() {
  contextp->timeInc(1);
  tfp->dump(contextp->time());
  tfp->close();
  if (bdb_sim_meta_path() == nullptr) {
    printf("The wave data has been saved to the FST file: %s\n", fst_path);
  }
  exit(0);
}

void ball_exec_once() {
  // posedge: clock=1, eval (FF outputs settle)
  top->clock = 1;
  top->eval();
  // SCU DPI-C functions (scu_uart_write, scu_sim_exit) are called automatically
  // from RTL
  contextp->timeInc(1);
  tfp->dump(contextp->time());
  sim_time++;

  // negedge: clock=0, eval
  top->clock = 0;
  step_and_dump_wave();
}

//================ main =====================//
int main(int argc, char *argv[]) {
  init_monitor(argc, argv);
  sim_init(argc, argv);
  bdb_mainloop();
  sim_exit();
}
