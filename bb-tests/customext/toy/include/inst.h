// #ifndef _INST_H
// #define _INST_H

// #include <cstdint>

// // ----------------------------- mvin --------------------------------------
// struct mvin_rs1_t {
//   uint64_t value;
//   explicit mvin_rs1_t(uint64_t val) : value(val) {}

//   uint32_t base_dram_addr() const { return (value >> 0) & 0xFFFFFFFF; }
// };

// struct mvin_rs2_t {
//   uint64_t value;
//   explicit mvin_rs2_t(uint64_t val) : value(val) {}

//   uint32_t base_sp_addr() const { return (value >> 0) & 0x3FFF; }
//   uint32_t iter() const { return (value >> 14) & 0x3FF; }
// };

// // ----------------------------- mvout --------------------------------------
// struct mvout_rs1_t {
//   uint64_t value;
//   explicit mvout_rs1_t(uint64_t val) : value(val) {}

//   uint32_t base_dram_addr() const { return (value >> 0) & 0xFFFFFFFF; }
// };

// struct mvout_rs2_t {
//   uint64_t value;
//   explicit mvout_rs2_t(uint64_t val) : value(val) {}

//   uint32_t base_sp_addr() const { return (value >> 0) & 0x3FFF; }
//   uint32_t iter() const { return (value >> 14) & 0x3FF; }
// };

// // ----------------------------- mul_warp16
// // --------------------------------------
// struct mul_warp16_rs1_t {
//   uint64_t value;
//   explicit mul_warp16_rs1_t(uint64_t val) : value(val) {}

//   uint32_t op1_spaddr() const { return (value >> 0) & 0x3FFF; }
//   uint32_t op2_spaddr() const { return (value >> 14) & 0x3FFF; }
// };

// struct mul_warp16_rs2_t {
//   uint64_t value;
//   explicit mul_warp16_rs2_t(uint64_t val) : value(val) {}

//   uint32_t wr_spaddr() const { return (value >> 0) & 0x3FFF; }
//   uint32_t iter() const { return (value >> 14) & 0x3FF; }
// };

// // ----------------------------- transpose ---------------------------------
// struct transpose_rs1_t {
//   uint64_t value;
//   explicit transpose_rs1_t(uint64_t val) : value(val) {}

//   uint32_t op_spaddr() const { return (value >> 0) & 0x3FFF; }
//   uint32_t wr_spaddr() const { return (value >> 14) & 0x3FFF; }
// };

// struct transpose_rs2_t {
//   uint64_t value;
//   explicit transpose_rs2_t(uint64_t val) : value(val) {}

//   uint32_t iter() const { return (value >> 14) & 0x7FFFFFFF; }
// };

// // ----------------------------- im2col ------------------------------------
// struct im2col_rs1_t {
//   uint64_t value;
//   explicit im2col_rs1_t(uint64_t val) : value(val) {}

//   uint32_t op_spaddr() const { return (value >> 0) & 0x3FFF; }
//   uint32_t wr_spaddr() const { return (value >> 14) & 0x3FFF; }
// };

// struct im2col_rs2_t {
//   uint64_t value;
//   explicit im2col_rs2_t(uint64_t val) : value(val) {}

//   uint32_t kcol() const { return (value >> 23) & 0x1; }       // 23:27
//   uint32_t krow() const { return (value >> 27) & 0x1; }       // 27:31
//   uint32_t inrow() const { return (value >> 31) & 0x3FFF; }   // 31:36
//   uint32_t incol() const { return (value >> 36) & 0xFFF; }    // 36:46
//   uint32_t startrow() const { return (value >> 46) & 0xFFF; } // 46:51
//   uint32_t startcol() const { return (value >> 51) & 0xFFF; } // 51:61
// };

// // ----------------------------- bbfp_mul -----------------------------------
// struct bbfp_mul_rs1_t {
//   uint64_t value;
//   explicit bbfp_mul_rs1_t(uint64_t val) : value(val) {}

//   uint32_t op1_spaddr() const { return (value >> 0) & 0x3FFF; }
//   uint32_t op2_spaddr() const { return (value >> 14) & 0x3FFF; }
// };

// struct bbfp_mul_rs2_t {
//   uint64_t value;
//   explicit bbfp_mul_rs2_t(uint64_t val) : value(val) {}

//   uint32_t wr_spaddr() const { return (value >> 0) & 0x3FFF; }
//   uint32_t iter() const { return (value >> 14) & 0x3FF; }
// };

// #endif // _INST_H
