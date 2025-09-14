#ifndef SPAD_H
#define SPAD_H

#include "bank.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t spad_addr(uint32_t bank, uint32_t row);
uint32_t spad_get_bank(uint32_t addr);
uint32_t spad_get_offset(uint32_t addr);
void spad_get_bank_offset(uint32_t addr, uint32_t *bank, uint32_t *offset);
uint32_t spad_get_bank_row_bytes(uint32_t bank);

#ifdef __cplusplus
}
#endif

#endif // SPAD_H
