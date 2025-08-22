#include "bdb.h"
#include "utils/macro.h"
#include "utils/debug.h"

#include "../build/obj_dir/VTestHarness___024root.h"

// #define MAX_SIM_TIME 50 最大仿真周期
vluint64_t sim_time = 0;

VerilatedContext* contextp = NULL;
VerilatedVcdC* tfp = NULL;
static VTestHarness* top;

int bb_step = 1; // 记录一共走了多少步，出错时抛出，方便单步调试到周围 

//================ SIM FUNCTION =====================//
void step_and_dump_wave() {
  top->eval();
  contextp->timeInc(1);
  tfp->dump(contextp->time());
  sim_time++;
}

void sim_init(int argc, char** argv) {
  contextp = new VerilatedContext;
  contextp->commandArgs(argc, argv);
  tfp = new VerilatedVcdC;
  top = new VTestHarness{contextp};

  contextp->traceEverOn(true);
  top->trace(tfp, 0);
  tfp->open(TOSTRING(CONFIG_VCD_PATH)); // 编译参数指定

  top->reset = 1; top->clock = 0; step_and_dump_wave();
  top->reset = 1; top->clock = 1; step_and_dump_wave();
  top->reset = 0; top->clock = 0; step_and_dump_wave();   


  // top->rootp->TestHarness__DOT__chiptop0__DOT__system__DOT__pbus__DOT__bootAddrReg = 0x80000000ULL;
} // 低电平复位

void ball_exec_once() {
  top->clock ^= 1; step_and_dump_wave();
  top->clock ^= 1; step_and_dump_wave();
  // top->rootp->TestHarness__DOT__chiptop0__DOT__system__DOT__pbus__DOT__bootAddrReg = 0x80000000ULL;
  if (top->io_success == 1) {
    printf("simulation success\n");
    exit(0);
  }
  // dump_gpr(); 
  // npc_step++;
  // printf("bootAddrReg = 0x%x\n", top->rootp->TestHarness__DOT__chiptop0__DOT__system__DOT__pbus__DOT__bootAddrReg);
} // 翻转两次走一条指令

void sim_exit() {
  contextp->timeInc(1);
  tfp->dump(contextp->time());
  tfp->close();
  printf("The wave data has been saved to the VCD file: %s\n", TOSTRING(CONFIG_VCD_PATH));
}

// void init_tet() {
//   while (cpu_npc.pc != MEM_BASE) { 
//     // printf("%ld\n", cpu_npc.pc); 
//     npc_exec_once(); 
//     // npc_step--;
//   } // pc先走拍到第一条指令执行结束
// }



//================ main =====================//
int main(int argc, char *argv[]) {
  sim_init(argc, argv);
  init_monitor(argc, argv);
  bdb_mainloop();
  sim_exit();
} 