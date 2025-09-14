#include "spad.h"

/**
 * SPAD (Scratchpad) 内存地址管理库
 *
 * 提供的 bank 和 row 到物理地址的转换
 */

// 运行时接口 - 纯 C 实现
uint32_t spad_addr(uint32_t bank, uint32_t row) {
  return (bank < BANK_NUM) ? bank_addr(&bank_configs[bank], row) : -1;
}

// 从spad地址获取bank编号
uint32_t spad_get_bank(uint32_t addr) {
  for (uint32_t bank = 0; bank < BANK_NUM; bank++) {
    uint32_t base = bank_configs[bank].base_addr_;
    uint32_t size = bank_configs[bank].row_num_;
    if (addr >= base && addr < base + size) {
      return bank;
    }
  }
  return -1; // 未找到对应的bank
}

// 从spad地址获取在bank内的偏移(row)
uint32_t spad_get_offset(uint32_t addr) {
  for (uint32_t bank = 0; bank < BANK_NUM; bank++) {
    uint32_t base = bank_configs[bank].base_addr_;
    uint32_t size = bank_configs[bank].row_num_;
    if (addr >= base && addr < base + size) {
      return addr - base;
    }
  }
  return -1; // 未找到对应的bank
}

// 从spad地址同时获取bank和offset
void spad_get_bank_offset(uint32_t addr, uint32_t *bank, uint32_t *offset) {
  for (uint32_t i = 0; i < BANK_NUM; i++) {
    uint32_t base = bank_configs[i].base_addr_;
    uint32_t size = bank_configs[i].row_num_;
    if (addr >= base && addr < base + size) {
      *bank = i;
      *offset = addr - base;
      return;
    }
  }
  *bank = -1;
  *offset = -1;
}

// 获取指定bank的行宽度（字节数）
uint32_t spad_get_bank_row_bytes(uint32_t bank) {
  if (bank >= BANK_NUM)
    return 0;
  return bank_configs[bank].elem_num_ * bank_configs[bank].elem_width_ / 8;
}
