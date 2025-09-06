#ifndef BUCKYBALL_H
#define BUCKYBALL_H

#include <stdint.h>

// String macros (from xcustom.h)
#define STR1(x) #x
#ifndef STR
#define STR(x) STR1(x)
#endif

#define CAT_(A, B) A##B
#define CAT(A, B) CAT_(A, B)

// Buckyball accelerator configuration
#define DIM 16                    // Matrix dimension
#define SPAD_ADDR_LEN 14         // Scratchpad address length in bits
#define MEM_ADDR_LEN 32          // Memory address length in bits
#define BANK_NUM 4               // Number of banks
#define BANK_ROWS 4096           // Rows per bank
#define SP_MATRICES ((BANK_NUM * BANK_ROWS) / DIM)  // Number of matrices in scratchpad

// RoCC instruction encoding
#define CUSTOM_3 0x7B            // Custom-3 opcode (0111 1011)

// Buckyball function codes (funct7 field)
#define BB_MVIN_FUNCT 24         // 0x18 - Move in function code
#define BB_MVOUT_FUNCT 25        // 0x19 - Move out function code  
#define BB_FENCE_FUNCT 31        // 0x1F - Fence function code
#define BB_MUL_FUNCT 32          // 0x20 - Matrix multiply function code
#define BB_IM2COL_FUNCT 33          // 0x21 - Matrix im2col function code
#define BB_FLUSH_FUNCT 7         // 0x07 - Flush function code
#define BB_BBFP_MUL_FUNCT 26     // 0x1A - BBFP matrix multiply function code
#define BB_MATMUL_WS_FUNCT 27    // 0x1B - Matrix multiply with warp16 function code
// Data type for matrix elements
typedef int8_t elem_t;
typedef int32_t result_t;

// Buckyball RoCC instruction macro (following xcustom.h style)
#define BUCKYBALL_INSTRUCTION_R_R(rs1, rs2, func7) \
  { \
    asm volatile( \
        ".insn r " STR(CUSTOM_3) ", " STR(0x3) ", " STR(func7) ", x0, %0, %1" \
        : \
        : "r"(rs1), "r"(rs2)); \
  }

#define BUCKYBALL_INSTRUCTION_FLUSH(func7) \
  { \
    asm volatile( \
        ".insn r " STR(CUSTOM_3) ", " STR(0x3) ", " STR(func7) ", x0, x0, x0" \
        : \
        :); \
  }

// Buckyball custom instructions using inline assembly

// Move data from DRAM to scratchpad
// mem_addr: DRAM address, sp_addr: scratchpad address, rows: number of rows to transfer
#define bb_mvin(mem_addr, sp_addr, rows) \
    do { \
        uint64_t rs1_val = (uint64_t)(mem_addr); \
        uint64_t rs2_val = ((rows) << SPAD_ADDR_LEN) | ((sp_addr) & ((1UL << SPAD_ADDR_LEN) - 1)); \
        BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, BB_MVIN_FUNCT); \
    } while(0)

// Move data from scratchpad to DRAM
// mem_addr: DRAM address, sp_addr: scratchpad address, rows: number of rows to transfer
#define bb_mvout(mem_addr, sp_addr, rows) \
    do { \
        uint64_t rs1_val = (uint64_t)(mem_addr); \
        uint64_t rs2_val = ((rows) << SPAD_ADDR_LEN) | ((sp_addr) & ((1UL << SPAD_ADDR_LEN) - 1)); \
        BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, BB_MVOUT_FUNCT); \
    } while(0)
#define bb_fence() \
    do { \
        uint64_t rs1_val = 0; \
        uint64_t rs2_val = 0; \
        BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, BB_FENCE_FUNCT); \
    } while(0)
// Matrix multiplication with warp16 pattern
// op1_addr: first operand scratchpad address, op2_addr: second operand scratchpad address  
// wr_addr: write result scratchpad address, iter: number of iterations
#define bb_mul_warp16(op1_addr, op2_addr, wr_addr, iter) \
    do { \
        uint64_t rs1_val = ((op2_addr) << SPAD_ADDR_LEN) | ((op1_addr) & ((1UL << SPAD_ADDR_LEN) - 1)); \
        uint64_t rs2_val = ((iter) << SPAD_ADDR_LEN) | ((wr_addr) & ((1UL << SPAD_ADDR_LEN) - 1)); \
        BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, BB_MUL_FUNCT); \
    } while(0)

#define bb_bbfp_mul(op1_addr, op2_addr, wr_addr, iter) \
do { \
    uint64_t rs1_val = ((op2_addr) << SPAD_ADDR_LEN) | ((op1_addr) & ((1UL << SPAD_ADDR_LEN) - 1)); \
    uint64_t rs2_val = ((iter) << SPAD_ADDR_LEN) | ((wr_addr) & ((1UL << SPAD_ADDR_LEN) - 1)); \
    BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, BB_BBFP_MUL_FUNCT); \
} while(0)

// OP1 Psum地址 OP2新的激活地址 wraddr写回的psum地址 iter迭代次数
#define bb_matmul_ws(op1_addr, op2_addr, wr_addr, iter) \
do { \
    uint64_t rs1_val = ((op2_addr) << SPAD_ADDR_LEN) | ((op1_addr) & ((1UL << SPAD_ADDR_LEN) - 1)); \
    uint64_t rs2_val = ((iter) << SPAD_ADDR_LEN) | ((wr_addr) & ((1UL << SPAD_ADDR_LEN) - 1)); \
    BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, BB_MATMUL_WS_FUNCT); \
} while(0)

#define bb_im2col(op1_addr, wr_addr, klen, iter) \
do { \
    uint64_t rs1_val = ((wr_addr) << SPAD_ADDR_LEN) | ((op1_addr) & ((1UL << SPAD_ADDR_LEN) - 1)); \
    uint64_t rs2_val = ( (klen) << (SPAD_ADDR_LEN + 10) | (iter) << SPAD_ADDR_LEN) ; \
    BUCKYBALL_INSTRUCTION_R_R(rs1_val, rs2_val, BB_IM2COL_FUNCT); \
} while(0)

// Flush accelerator
#define bb_flush() \
    BUCKYBALL_INSTRUCTION_FLUSH(BB_FLUSH_FUNCT)

#define MULTICORE_INIT(hart_id) \
  __attribute__((constructor)) \
  static void _multicore_init() { \
    multicore(hart_id); \
  }

static inline void multicore(int target_hart_id) {
  int hart_id;
  asm volatile("csrr %0, mhartid" : "=r"(hart_id));
  
  if (hart_id != target_hart_id) {
    while (1) {
      asm volatile("wfi");  // Wait for interrupt
    }
  }
  // If hart_id == target_hart_id, continue execution
}

// Utility functions
void print_u32_matrix(const char* name, result_t* matrix, int rows, int cols);
void print_u8_matrix(const char* name, elem_t* matrix, int rows, int cols);

void init_u8_random_matrix(elem_t* matrix, int rows, int cols, int seed);
void init_u32_random_matrix(result_t* matrix, int rows, int cols, int seed);

int compare_u8_matrices(elem_t* a, elem_t* b, int rows, int cols);
int compare_u32_matrices(result_t* a, result_t* b, int rows, int cols);
int compare_u32_matrices_with_tolerance(result_t *a, result_t *b, 
                                       int rows, int cols, double tolerance);
void clear_u32_matrix(result_t* matrix, int rows, int cols);
void clear_u8_matrix(elem_t* matrix, int rows, int cols);

void init_ones_matrix(elem_t* matrix, int rows, int cols);
void init_identity_matrix(elem_t* matrix, int size);
void init_row_vector(elem_t* matrix, int cols, elem_t value);
void init_col_vector(elem_t* matrix, int rows, elem_t value);
void init_random_matrix(elem_t* matrix, int rows, int cols, int seed);
void init_bbfp_random_matrix(elem_t* matrix, int rows, int cols, int seed);
void init_sequence_matrix(elem_t* matrix, int rows, int cols);

/* 矩阵运算函数 */
void transpose_u8_matrix(elem_t* src, elem_t* dst, int rows, int cols);
void transpose_u32_matrix(result_t* src, result_t* dst, int rows, int cols);
void cpu_matmul(elem_t* a, elem_t* b, result_t* c, int rows, int cols, int inner);
#endif