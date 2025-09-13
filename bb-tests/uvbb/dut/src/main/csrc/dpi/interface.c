#include "dpi/interface.h"
#include "utils/debug.h"
#include <stdio.h>
#include <string.h>

// SRAM读写接口实现
void sram_read(int bank_id, int addr, unsigned char data[16]) {
  // TODO: 实现SRAM读取逻辑
  memset(data, 0, 16);
  Log("SRAM读取: bank=%d, addr=0x%x", bank_id, addr);
}

void sram_write(int bank_id, int addr, unsigned char data[16], int mask) {
  // TODO: 实现SRAM写入逻辑
  Log("SRAM写入: bank=%d, addr=0x%x, mask=0x%x", bank_id, addr, mask);
}

// 累加器读写接口实现
void acc_read(int bank_id, int addr, int data[4]) {
  // TODO: 实现累加器读取逻辑
  memset(data, 0, 16);
  Log("ACC读取: bank=%d, addr=0x%x", bank_id, addr);
}

void acc_write(int bank_id, int addr, int data[4], int mask) {
  // TODO: 实现累加器写入逻辑
  Log("ACC写入: bank=%d, addr=0x%x, mask=0x%x", bank_id, addr, mask);
}

// 命令接口实现
void cmd_request(unsigned char valid, unsigned char ready, int bid, int iter,
                 long long special, int rob_id) {
  if (valid && ready) {
    Log("命令请求: bid=%d, iter=%d, special=0x%llx, rob_id=%d", bid, iter,
        special, rob_id);
  }
}

void cmd_response(unsigned char valid, unsigned char ready, int rob_id) {
  if (valid && ready) {
    Log("命令响应: rob_id=%d", rob_id);
  }
}
