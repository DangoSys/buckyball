#ifndef _CORE_H_
#define _CORE_H_

// 暂时简化，等Verilator集成后恢复
// #include "verilated.h"
// #include "verilated_vcd_c.h"
// #include "VBallTop.h"
#include <cstdio>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

// 仿真控制接口
void sim_init(int argc, char **argv);
void sim_exit(int status);
void step_and_dump_wave();

// 主要DUT接口函数 - 必须使用extern "C"避免名称修饰
void ball_exec_once();

// 仿真循环
void mainloop();

#ifdef __cplusplus
}
#endif

#endif // _CORE_H_
