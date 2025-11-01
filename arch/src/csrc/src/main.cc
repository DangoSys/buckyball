#include "bdb.h"
#include "utils/debug.h"
#include "utils/macro.h"

// #include "../build/obj_dir/VTestHarness___024root.h"

// #define MAX_SIM_TIME 50 Maximum simulation cycles
vluint64_t sim_time = 0;

VerilatedContext *contextp = NULL;
// VerilatedVcdC *tfp = NULL;
VerilatedFstC *tfp = NULL;
static VTestHarness *top;

// Record how many steps taken, useful for debugging when errors occur
int bb_step = 1;

//================ SIM FUNCTION =====================//
void step_and_dump_wave() {
  top->eval();
  contextp->timeInc(1);
  tfp->dump(contextp->time());
  sim_time++;
}

void sim_init(int argc, char **argv) {
  contextp = new VerilatedContext;
  contextp->commandArgs(argc, argv);
  // tfp = new VerilatedVcdC;
  tfp = new VerilatedFstC;
  top = new VTestHarness{contextp};

  contextp->traceEverOn(true);
  top->trace(tfp, 0);

  // tfp->open(vcd_path);
  // Log("The waveform will be saved to the VCD file: %s", vcd_path);
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

  // top->rootp->TestHarness__DOT__chiptop0__DOT__system__DOT__pbus__DOT__bootAddrReg
  // = 0x80000000ULL;
  // Low-level reset
}

void sim_exit() {
  contextp->timeInc(1);
  tfp->dump(contextp->time());
  tfp->close();
  // printf("The wave data has been saved to the VCD file: %s\n", vcd_path);
  printf("The wave data has been saved to the FST file: %s\n", fst_path);
  exit(0);
}

void ball_exec_once() {
  top->clock ^= 1;
  step_and_dump_wave();
  top->clock ^= 1;
  step_and_dump_wave();
  // top->rootp->TestHarness__DOT__chiptop0__DOT__system__DOT__pbus__DOT__bootAddrReg
  // = 0x80000000ULL;
  if (top->io_success == 1) {
    printf("simulation success\n");
    sim_exit();
  }
  // dump_gpr();
  // npc_step++;
  // printf("bootAddrReg = 0x%x\n",
  // top->rootp->TestHarness__DOT__chiptop0__DOT__system__DOT__pbus__DOT__bootAddrReg);
  // Toggle twice to execute one instruction
}

// void init_tet() {
//   while (cpu_npc.pc != MEM_BASE) {
//     // printf("%ld\n", cpu_npc.pc);
//     npc_exec_once();
//     // npc_step--;
//   } // PC advances until first instruction execution completes
// }

//================ main =====================//
int main(int argc, char *argv[]) {
  // Parse parameters here, including VCD path
  init_monitor(argc, argv);
  sim_init(argc, argv);
  bdb_mainloop();
  sim_exit();
}
