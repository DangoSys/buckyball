#ifndef _INST_H
#define _INST_H

#include <cstdint>

// ----------------------------- mvin --------------------------------------
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

// ----------------------------- mvout --------------------------------------
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

// ----------------------------- mul_warp16
// --------------------------------------
struct mul_warp16_rs1_t {
  uint64_t value;
  explicit mul_warp16_rs1_t(uint64_t val) : value(val) {}

  uint32_t op1_spaddr() const { return (value >> 0) & 0x3FFF; }
  uint32_t op2_spaddr() const { return (value >> 14) & 0x3FFF; }
};

struct mul_warp16_rs2_t {
  uint64_t value;
  explicit mul_warp16_rs2_t(uint64_t val) : value(val) {}

  uint32_t wr_spaddr() const { return (value >> 0) & 0x3FFF; }
  uint32_t iter() const { return (value >> 14) & 0x3FF; }
};

// ----------------------------- scatter_mvin
// --------------------------------------
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

// ----------------------------- sparse_mul
// --------------------------------------
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

// ----------------------------- bbfp_mul --------------------------------------
struct bbfp_mul_rs1_t {
  uint64_t value;
  explicit bbfp_mul_rs1_t(uint64_t val) : value(val) {}

  uint32_t op1_spaddr() const { return (value >> 0) & 0x3FFF; }
  uint32_t op2_spaddr() const { return (value >> 14) & 0x3FFF; }
};

struct bbfp_mul_rs2_t {
  uint64_t value;
  explicit bbfp_mul_rs2_t(uint64_t val) : value(val) {}

  uint32_t wr_spaddr() const { return (value >> 0) & 0x3FFF; }
  uint32_t iter() const { return (value >> 14) & 0x3FF; }
};

#endif // _INST_H
