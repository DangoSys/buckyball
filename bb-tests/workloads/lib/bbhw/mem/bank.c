#include "bank.h"

const BankConfig bank_configs[BANK_NUM] = {
    // SPAD area: bank 0-3, address 0-16383
    // bank 0: SPAD0, rows 0-4095
    {"spad0", 0, 4096, 16, 8},
    // bank 1: SPAD1, rows 4096-8191
    {"spad1", 4096, 4096, 16, 8},
    // bank 2: SPAD2, rows 8192-12287
    {"spad2", 8192, 4096, 16, 8},
    // bank 3: SPAD3, rows 12288-16383
    {"spad3", 12288, 4096, 16, 8},

    // ACC area: bank 4+, starting at address 16384
    // (each ACC bank actually has 1024 rows, but address is aligned to 4096)
    // bank 4: ACC0, rows 16384-17407 (uses first 1024 rows), accumulator uses
    // 32 bits
    {"acc0", 16384, 1024, 16, 32},
};
