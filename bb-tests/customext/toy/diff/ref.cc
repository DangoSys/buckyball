
#include "difftest.h"
#include "toy.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

// 全局difftest状态
static bool difftest_enabled = false;
static bool difftest_running = false;

// DUT库相关
static void *dut_handle = nullptr;
static void (*ball_exec_once)() = nullptr;
static bool dut_init = false;

void difftest_init() {
  difftest_enabled = true;
  printf("Difftest初始化完成\n");
}

void difftest_exec(uint64_t n) {
  // 空实现
}

void difftest_cleanup() {
  difftest_enabled = false;
  printf("Difftest清理完成\n");
}

void difftest_start(bool enable) {
  if (!enable || !difftest_enabled)
    return;

  printf("=== 开始difftest ===\n");
  difftest_running = true;
}

void difftest_end(bool enable) {
  if (!enable || !difftest_enabled)
    return;

  printf("=== 结束difftest ===\n");
  difftest_running = false;
}
