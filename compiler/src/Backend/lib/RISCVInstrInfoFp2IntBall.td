//===- RISCVInstrInfoFp2IntBall.td - Fp2IntBall instructions -------------===//
//
// Fp2IntBall instructions (fp2int)
//
//===----------------------------------------------------------------------===//

// bb_fp2int: funct7=51 (0b0110011)
let hasSideEffects = 1, mayLoad = 1, mayStore = 1, isCodeGenOnly = 1,
    Predicates = [HasBuddyExt] in
def BB_FP2INT : RVInstR<0b0110011, 0b011, OPC_CUSTOM_3, (outs),
                    (ins GPR:$rs1, GPR:$rs2), "bb_fp2int", "$rs1, $rs2"> {
  let rd = 0;
}

// Intrinsic pattern
let Predicates = [HasBuddyExt] in
def : Pat<(int_riscv_bb_fp2int GPR:$rs1, GPR:$rs2), (BB_FP2INT GPR:$rs1, GPR:$rs2)>;
