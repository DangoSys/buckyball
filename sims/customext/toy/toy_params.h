#ifndef TOY_PARAMS_H
#define TOY_PARAMS_H

#include <limits.h>
#include <stdint.h>

#define XCUSTOM_ACC 3
#define DIM 16
#define MEM_ADDR_LEN 32
#define SPAD_ADDR_LEN 14
#define BANK_NUM 4
#define BANK_ROWS 4096

typedef int8_t elem_t;
static const elem_t elem_t_max = 127;
static const elem_t elem_t_min = -128;

#define row_align(blocks)                                                      \
  __attribute__((aligned(blocks * DIM * sizeof(elem_t))))

// ----------------------- NVR --------------------------------------

struct mvin_rs1_t {
  uint64_t value;
  explicit mvin_rs1_t(uint64_t val) : value(val) {}

  uint32_t base_dram_addr() const { return (value >> 0) & 0xFFFFFFFF; }
};

struct mvin_rs2_t {
  uint64_t value;
  explicit mvin_rs2_t(uint64_t val) : value(val) {}

  uint32_t base_sp_addr() const { return (value >> 0) & 0x3FFF; }
  uint32_t rows() const { return (value >> 14) & 0x3FF; }
};

struct mvout_rs1_t {
  uint64_t value;
  explicit mvout_rs1_t(uint64_t val) : value(val) {}

  uint32_t base_dram_addr() const { return (value >> 0) & 0xFFFFFFFF; }
};

struct mvout_rs2_t {
  uint64_t value;
  explicit mvout_rs2_t(uint64_t val) : value(val) {}

  uint32_t base_sp_addr() const { return (value >> 0) & 0x3FFF; }
  uint32_t rows() const { return (value >> 14) & 0x3FF; }
};

struct mul_warp16_rs1_t {
  uint64_t value;
  explicit mul_warp16_rs1_t(uint64_t val) : value(val) {}

  uint32_t op1_addr() const { return (value >> 0) & 0x3FFF; }
  uint32_t op2_addr() const { return (value >> 14) & 0x3FFF; }
};

struct mul_warp16_rs2_t {
  uint64_t value;
  explicit mul_warp16_rs2_t(uint64_t val) : value(val) {}

  uint32_t wr_addr() const { return (value >> 0) & 0x3FFF; }
  uint32_t iter() const { return (value >> 14) & 0x3FF; }
};

struct scatter_mvin_rs1_t {
  uint64_t value;
  explicit scatter_mvin_rs1_t(uint64_t val) : value(val) {}

  uint32_t base_dram_addr() const { return (value >> 0) & 0xFFFFFFFF; }
};

struct scatter_mvin_rs2_t {
  uint64_t value;
  explicit scatter_mvin_rs2_t(uint64_t val) : value(val) {}

  uint32_t rf_bank() const { return (value >> 0) & 0x1; }
  uint32_t count() const { return (value >> 1) & 0x7FFFFFFF; }
};

struct sparse_mul_rs1_t {
  uint64_t value;
  explicit sparse_mul_rs1_t(uint64_t val) : value(val) {}

  uint32_t A_addr() const { return (value >> 0) & 0x3FFF; }
  uint32_t B_addr() const { return (value >> 14) & 0x3FFF; }
};

struct sparse_mul_rs2_t {
  uint64_t value;
  explicit sparse_mul_rs2_t(uint64_t val) : value(val) {}

  uint32_t row_rf_bank() const { return (value >> 0) & 0x1; }
  uint32_t col_rf_bank() const { return (value >> 1) & 0x1; }
  uint32_t C_addr() const { return (value >> 2) & 0x3FFF; }
  uint32_t nnz() const { return (value >> 16) & 0xFFF; }
};

// ------------------------------------------------------------------

#endif // TOY_PARAMS_H
