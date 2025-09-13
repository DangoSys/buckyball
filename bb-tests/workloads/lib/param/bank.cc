#include "bank.h"

const BankConfig bank_configs[BANK_NUM] = {
    {"spad0", 0, 1024, 16, 8},    // bank 0: SPAD0, 行0-1023
    {"spad1", 1024, 1024, 16, 8}, // bank 1: SPAD1, 行1024-2047
    {"spad2", 2048, 1024, 16, 8}, // bank 2: SPAD2, 行2048-3071
    {"spad3", 3072, 1024, 16, 8}, // bank 3: SPAD3, 行3072-4095
    {"acc0", 4096, 1024, 16, 32}, // bank 4: ACC0,  行4096-5119, 累加器用32位
};
