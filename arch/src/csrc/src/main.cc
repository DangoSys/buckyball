#include "bdb.h"
#include "utils/debug.h"
#include "utils/macro.h"

#ifdef COSIM
#include "monitor/cosim.h"
#endif

#include <csignal>
#include <cstdlib>

vluint64_t sim_time = 0;
VerilatedContext *contextp = NULL;
VerilatedFstC *tfp = NULL;

#ifdef COSIM
static VToyBuckyball *top;
static CosimServer *cosim_server = NULL;
#else
static VTestHarness *top;
#endif

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

#if VM_COVERAGE
  atexit(coverage_atexit);
  signal(SIGTERM, coverage_signal_handler);
  signal(SIGINT, coverage_signal_handler);
#endif

#ifdef COSIM
  // Initialize COSIM socket server
  cosim_server = new CosimServer(top);
  if (!cosim_server->init()) {
    panic("Failed to initialize COSIM server");
  }
  Log("COSIM mode: waiting for Bebop connection...");
#endif
}

void sim_exit() {
  contextp->timeInc(1);
  tfp->dump(contextp->time());
  tfp->close();
  printf("The wave data has been saved to the FST file: %s\n", fst_path);

#ifdef COSIM
  if (cosim_server) {
    cosim_server->shutdown();
    delete cosim_server;
  }
#endif

  exit(0);
}

void ball_exec_once() {
#ifdef COSIM
  // Update COSIM server (handle socket I/O and drive DUT signals)
  if (cosim_server) {
    cosim_server->update();
  }
#endif

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
