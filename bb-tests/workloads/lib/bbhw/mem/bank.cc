#include "bank.h"

const BankConfig bank_configs[BANK_NUM] = {
    {"spad0", 0, 512, 16, 8},     // bank 0: SPAD0, 行0-1023
    {"spad1", 512, 512, 16, 8},   // bank 1: SPAD1, 行1024-2047
    {"acc0", 1024, 1024, 16, 32}, // bank 2: ACC0,  行4096-5119, 累加器用32位
};
