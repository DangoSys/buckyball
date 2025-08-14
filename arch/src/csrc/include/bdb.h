#ifndef _BDB_H_
#define _BDB_H_

// DPI-C
#include "verilated_dpi.h"
// #include "Vtop__Dpi.h"
#include "svdpi.h"
// verilator
#include "verilated.h"
#include "verilated_vcd_c.h"
#include "VTestHarness.h"


// ================ DataType ====================

// ================ RISCV CPU ===================

// ================ BDB Config ===================
void init_monitor(int argc, char *argv[]);
void bdb_mainloop();
void ball_exec_once();
void bdb_set_batch_mode(); 



#endif // _BDB_H_