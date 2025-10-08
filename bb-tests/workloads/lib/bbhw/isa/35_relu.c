#include "isa.h"

// =========================== for simulator ===========================
const InstructionConfig relu_config = {
    .rs1_fields = (BitFieldConfig[]){{"op1_spaddr", 0, 13},
                                     {NULL, 0, 0}},
    .rs2_fields = (BitFieldConfig[]){
        {"wr_spaddr", 0, 13}, {"iter", 14, 23}, {NULL, 0, 0}}};

// =========================== for CTest ===========================
#define RELU_ENCODE_RS1(op1_addr) ENCODE_FIELD(op1_addr, 0, 14)

#define RELU_ENCODE_RS2(wr_addr, iter) \
    (ENCODE_FIELD(wr_addr, 0, 14) | ENCODE_FIELD(iter, 14, 10))

// RELU指令低级实现
#ifndef __x86_64__
#define RELU_RAW(rs1, rs2)                                        \
    asm volatile(".insn r " STR(CUSTOM_3) ", 0x3, 35, x0, %0, %1" \
                 :                                                \
                 : "r"(rs1),                                      \
                   "r"(rs2)                                       \
                 : "memory")
#else
#define RELU_RAW(rs1, rs2) /* x86平台下不执行RISC-V指令 */
#endif

// RELU指令高级API实现
void bb_relu(uint32_t op1_addr, uint32_t wr_addr, uint32_t iter)
{
    uint32_t rs1_val = RELU_ENCODE_RS1(op1_addr);
    uint32_t rs2_val = RELU_ENCODE_RS2(wr_addr, iter);
    RELU_RAW(rs1_val, rs2_val);
}