#ifndef _DIFFTEST_H_
#define _DIFFTEST_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Difftest接口函数
void difftest_init();
void difftest_exec(uint64_t n);
void difftest_cleanup();

// difftest触发接口
void difftest_start(bool enable);
void difftest_end(bool enable);

#ifdef __cplusplus
}
#endif

#endif
