#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig conv_config = {
    .rs1_fields = (BitFieldConfig[]){{"ifmap_spaddr", 0, 14},
                                     {"weight_spaddr", 15, 29},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"ofmap_spaddr", 0, 14},
        {"iter", 15, 24},
        // special field is 40 bits: rs2(63, spAddrLen + 10) = rs2(63, 24) = 40
        // bits Encode: in_height[15:0], in_width[31:16], kernel_h[39:32],
        // kernel_w[47:40] but only 40 bits available Adjust: use
        // in_height[15:0], in_width[31:16], kernel_h[39:32], kernel_w same as
        // kernel_h
        {"in_height", 25, 40},
        {"in_width", 41, 56},
        {"kernel_h", 57, 64},
        {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define CONV_ENCODE_RS1(ifmap_addr, weight_addr)                               \
  (ENCODE_FIELD(ifmap_addr, 0, 15) | ENCODE_FIELD(weight_addr, 15, 15))

// Note: special field is only 40 bits (rs2[63:24])
// DomainDecoder extracts rs2(63, spAddrLen + 10) = rs2(63, 24) for special
// So special[39:0] = rs2[63:24]
// Encode in special: in_height[15:0] = special[15:0], in_width[15:0] =
// special[31:16], kernel_h[7:0] = special[39:32] kernel_w is assumed to equal
// kernel_h for simplicity
#define CONV_ENCODE_RS2(ofmap_addr, iter, in_height, in_width, kernel_h,       \
                        kernel_w)                                              \
  (ENCODE_FIELD(ofmap_addr, 0, 15) | ENCODE_FIELD(iter, 15, 10) |              \
   ENCODE_FIELD(in_height, 24, 16) | ENCODE_FIELD(in_width, 40, 16) |          \
   ENCODE_FIELD(kernel_h, 56, 8))

// CONV instruction low-level implementation
#ifndef __x86_64__
#define CONV_RAW(rs1, rs2)                                                     \
  asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 43, x0, %0, %1"                \
               :                                                               \
               : "r"(rs1), "r"(rs2)                                            \
               : "memory")
#else
// Do not execute RISC-V instructions on x86 platform
#define CONV_RAW(rs1, rs2)
#endif

// CONV instruction high-level API implementation
void bb_conv(uint32_t ifmap_addr, uint32_t weight_addr, uint32_t ofmap_addr,
             uint32_t iter, uint32_t in_height, uint32_t in_width,
             uint32_t kernel_h, uint32_t kernel_w) {
  uint64_t rs1_val = CONV_ENCODE_RS1(ifmap_addr, weight_addr);
  uint64_t rs2_val = CONV_ENCODE_RS2(ofmap_addr, iter, in_height, in_width,
                                     kernel_h, kernel_w);
  CONV_RAW(rs1_val, rs2_val);
}
