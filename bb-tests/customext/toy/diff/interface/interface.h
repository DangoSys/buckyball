#ifndef INTERFACE_H
#define INTERFACE_H

#include <cstdint>

typedef struct {
  bool enabled;
  bool in_vecunit_mode;
  uint64_t cycle_count;
  int current_iter;
  int total_iters;
} difftest_state_t;

// VecUnit状态同步结构
typedef struct {
  uint32_t cmd_bid;
  uint32_t cmd_iter;
  uint64_t cmd_special;
  uint32_t rob_id;
  bool cmd_valid;
  bool cmd_ready;
} vecunit_cmd_t;

typedef struct {
  uint8_t data[16]; // SRAM数据 (128位)
  uint32_t addr;
  uint8_t mask;
  bool valid;
} sram_data_t;

typedef struct {
  int32_t data[4]; // ACC数据 (4个32位)
  uint32_t addr;
  uint8_t mask;
  bool valid;
} acc_data_t;

#endif // INTERFACE_H
