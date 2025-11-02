#ifndef BUCKYBALL_ISA_H
#define BUCKYBALL_ISA_H

#include <stddef.h>
#include <stdint.h>

/* Pure C implementation - no C++ linkage needed */

// Data type for matrix elements
typedef int8_t elem_t;
typedef int32_t result_t;

// Custom instruction opcodes
#define CUSTOM_3 0x7b
// String macros (from xcustom.h)
#define STR1(x) #x
#ifndef STR
#define STR(x) STR1(x)
#endif

// Generic field encoding macro
#define ENCODE_FIELD(value, start_bit, width)                                  \
  (((value) & ((1ULL << (width)) - 1)) << (start_bit))

// Bit field configuration structure
typedef struct {
  const char *name;   // Field name (NULL indicates end of array)
  uint32_t start_bit; // Start bit
  uint32_t end_bit;   // End bit (inclusive)
} BitFieldConfig;

// Instruction type enum - directly uses func7 values
typedef enum {
  MVIN_FUNC7 = 24,       // 0x18 - Move in function code
  MVOUT_FUNC7 = 25,      // 0x19 - Move out function code
  FENCE_FUNC7 = 31,      // 0x1F - Fence function code
  MUL_WARP16_FUNC7 = 32, // 0x20 - Matrix multiply function code
  IM2COL_FUNC7 = 33,     // 0x21 - Matrix im2col function code
  TRANSPOSE_FUNC7 = 34,  // 0x22 - Matrix transpose function code
  FLUSH_FUNC7 = 7,       // 0x07 - Flush function code
  BBFP_MUL_FUNC7 = 26,   // 0x1A - BBFP matrix multiply function code
  MATMUL_WS_FUNC7 = 27,  // 0x1B - Matrix multiply with warp16 function code
  RELU_FUNC7 = 38,       // 0x26 - ReLU activation function code
  BBUS_CONFIG_FUNC7 = 39 // 0x27 - BBUS configuration function code
} InstructionType;

// Instruction configuration structure (for simulator)
typedef struct {
  // Field configuration for rs1 register (terminated by NULL name)
  const BitFieldConfig *rs1_fields;
  // Field configuration for rs2 register (terminated by NULL name)
  const BitFieldConfig *rs2_fields;
} InstructionConfig;

// Generic field access functions (for simulator)
uint32_t get_bbinst_field(uint64_t value, const char *field_name,
                          const BitFieldConfig *config);
void set_bbinst_field(uint64_t *value, const char *field_name,
                      uint32_t field_value, const BitFieldConfig *config);

// High-level API (for CTest)

void bb_mvin(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter,
             uint32_t col_stride);
void bb_mvout(uint64_t mem_addr, uint32_t sp_addr, uint32_t iter,
              uint32_t stride);
void bb_fence(void);
void bb_mul_warp16(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                   uint32_t iter, uint32_t mode);
void bb_bbfp_mul(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                 uint32_t iter);
void bb_matmul_ws(uint32_t op1_addr, uint32_t op2_addr, uint32_t wr_addr,
                  uint32_t iter);
void bb_im2col(uint32_t op1_addr, uint32_t wr_addr, uint32_t krow,
               uint32_t kcol, uint32_t inrow, uint32_t incol, uint32_t startrow,
               uint32_t startcol);
void bb_transpose(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter,
                  uint32_t mode);
void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter);
void bb_bbus_config(uint32_t src_bid, uint32_t dst_bid, uint64_t enable);
void bb_flush(void);

// Get instruction configuration by func7
const InstructionConfig *config(InstructionType func7);

/* End of pure C header */

#endif // BUCKYBALL_ISA_H
