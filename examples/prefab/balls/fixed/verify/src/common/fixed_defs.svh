import "DPI-C" function longint unsigned fixed_case_word(input int unsigned index, input int unsigned bank, input int unsigned row, input int unsigned half_word);
localparam int FIXED_THREAD_FUNCT7 = `FIXED_FUNCT7;
localparam int FIXED_MAX_ROWS = 16;
localparam int FIXED_NUM_CASES = 108;
localparam int FIXED_TIMEOUT_CYCLES = 20000;
import "DPI-C" function longint unsigned fixed_bemu_cycles(
    input longint unsigned iter, input longint unsigned rs2, input int a[4096], input int b[4096]);
