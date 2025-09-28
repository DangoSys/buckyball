#ifndef _BDB_H_
#define _BDB_H_

// DPI-C
#include "verilated_dpi.h"
// #include "Vtop__Dpi.h"
#include "svdpi.h"
// verilator
#include "verilated.h"
// #include "verilated_vcd_c.h"
#include "VTestHarness.h"
#include "verilated_fst_c.h"

// ================ DataType ====================

// ================ RISCV CPU ===================

// ================ BDB Config ===================
extern const char *vcd_path; // VCD文件路径
extern const char *log_path; // 日志文件路径
extern const char *fst_path; // FST文件路径

void init_monitor(int argc, char *argv[]);
void bdb_mainloop();
void ball_exec_once();
void bdb_set_batch_mode();

#endif // _BDB_H_
