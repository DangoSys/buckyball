//===- RISCVInstrInfoInt2FpBall.td - Int2FpBall instructions -------------===//
//
// Int2FpBall instructions (int2fp)
//
//===----------------------------------------------------------------------===//

// bb_int2fp: funct7=52 (0b0110100)
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_INT2FP : RVInstR<0b0110100, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_int2fp", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic pattern
let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_int2fp GPR:$rs1, GPR:$rs2), (BB_INT2FP GPR:$rs1, GPR:$rs2)>;
