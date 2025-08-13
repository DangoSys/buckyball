#ifndef _TET_H_
#define _TET_H_

// DPI-C
#include "verilated_dpi.h"
// #include "Vtop__Dpi.h"
#include "svdpi.h"
// verilator
#include "verilated.h"
#include "verilated_vcd_c.h"
#include "VChipTop.h"


// ================ DataType ====================
#include "input/input.h"
#include "utils/display.h"
#include "utils/config.h"


// ================ RISCV CPU ===================

// ================ SDB Config ===================
void init_monitor(int argc, char *argv[]);
void sdb_mainloop();
void tet_exec_once();
void sdb_set_batch_mode(); 

// int read_fp16_array(const char *filename, fp16_t array[datain_lines][datain_bandwidth]);
// ================ Ref ===================

// extern uint8_t softmax_input[16][16];
// void display_result(uint8_t (*dut_matrix)[16], double (*ref_matrix)[16]);
// void share_exp(fp16_t (*matrix_in)[datain_bandwidth], fp16_t (*matrix_out)[datain_bandwidth]);
// void softmax(uint8_t (*softmax_matrix)[16]);


#endif