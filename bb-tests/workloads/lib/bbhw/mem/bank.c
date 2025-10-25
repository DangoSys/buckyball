#include "bank.h"

const BankConfig bank_configs[BANK_NUM] = {
    // SPAD区域: bank 0-3, 地址0-16383
    {"spad0", 0, 4096, 16, 8},     // bank 0: SPAD0, 行0-4095
    {"spad1", 4096, 4096, 16, 8},  // bank 1: SPAD1, 行4096-8191
    {"spad2", 8192, 4096, 16, 8},  // bank 2: SPAD2, 行8192-12287
    {"spad3", 12288, 4096, 16, 8}, // bank 3: SPAD3, 行12288-16383

    // ACC区域: bank 4+, 地址16384开始（每个ACC
    // bank实际1024行，但地址按4096对齐）
    {"acc0", 16384, 1024, 16,
     32}, // bank 4: ACC0, 行16384-17407 (使用前1024行), 累加器用32位
};
