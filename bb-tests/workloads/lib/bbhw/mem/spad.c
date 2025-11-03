#include "spad.h"

/**
 * SPAD (Scratchpad) memory address management library
 *
 * Provides conversion from bank and row to physical address
 */

// Runtime interface - pure C implementation
uint32_t spad_addr(uint32_t bank, uint32_t row) {
  return (bank < BANK_NUM) ? bank_addr(&bank_configs[bank], row) : -1;
}

// Get bank number from spad address
uint32_t spad_get_bank(uint32_t addr) {
  for (uint32_t bank = 0; bank < BANK_NUM; bank++) {
    uint32_t base = bank_configs[bank].base_addr_;
    uint32_t size = bank_configs[bank].row_num_;
    if (addr >= base && addr < base + size) {
      return bank;
    }
  }
  // Bank not found
  return -1;
}

// Get offset (row) within bank from spad address
uint32_t spad_get_offset(uint32_t addr) {
  for (uint32_t bank = 0; bank < BANK_NUM; bank++) {
    uint32_t base = bank_configs[bank].base_addr_;
    uint32_t size = bank_configs[bank].row_num_;
    if (addr >= base && addr < base + size) {
      return addr - base;
    }
  }
  // Bank not found
  return -1;
}

// Get both bank and offset from spad address
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

// Get row width (bytes) of specified bank
uint32_t spad_get_bank_row_bytes(uint32_t bank) {
  if (bank >= BANK_NUM)
    return 0;
  return bank_configs[bank].elem_num_ * bank_configs[bank].elem_width_ / 8;
}
