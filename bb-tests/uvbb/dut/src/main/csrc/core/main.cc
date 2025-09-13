#include "core/core.h"
#include "utils/debug.h"

// 声明monitor函数
extern "C" void init_monitor(int argc, char *argv[]);

// 全局变量
static int sim_time = 0;
static int bb_step = 1;

// 暂时用简单的模拟，等Verilator集成后替换
// VerilatedContext* contextp = nullptr;
// VerilatedVcdC* tfp = nullptr;
// static VBallTop* top = nullptr;

// 简化的仿真控制函数（暂时模拟）
void step_and_dump_wave() {
  // 模拟一个仿真步骤
  sim_time++;
}

void sim_init(int argc, char **argv) {
  // 简化的初始化
  (void)argc;
  (void)argv; // 避免未使用参数警告
  sim_time = 0;
  bb_step = 1;
  printf("DUT仿真初始化完成\n");
}

void sim_exit(int status) {
  printf("DUT仿真退出，状态: %d\n", status);
  exit(status);
}

extern "C" void ball_exec_once() {
  // 模拟执行一个时钟周期
  step_and_dump_wave();
  step_and_dump_wave(); // 两个边沿
  bb_step++;
}

void mainloop() {
  while (1) {
    ball_exec_once();
  }
}

// 主程序入口
int main(int argc, char *argv[]) {
  init_monitor(argc, argv);
  sim_init(argc, argv);
  mainloop();
  sim_exit(0);
}
