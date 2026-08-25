//===-- GemminiBall.td - Gemmini Ball operations ------------------------===//

#ifndef GEMMINI_BALL_OPS
#define GEMMINI_BALL_OPS

include "Buckyball.td"

def GemminiInstructionOp : Buckyball_Op<"gemmini"> {
  let summary = "Gemmini command encoded for the target ISA";
  let description = [{
    Emits one Gemmini command. mnemonic must name an ISA entry enabled by the
    selected target, and rs1/rs2 are the exact command operands defined by
    that entry. This covers GEMMINI_CONFIG, GEMMINI_PRELOAD,
    GEMMINI_COMPUTE_PRELOADED, GEMMINI_COMPUTE_ACCUMULATED, GEMMINI_LOOP_WS,
    and GEMMINI_LOOP_CONV_WS without hard-coding funct7 values in the IR.
  }];
  let arguments = (ins StrAttr:$mnemonic, I64:$rs1, I64:$rs2);
  let assemblyFormat = "$mnemonic $rs1 `,` $rs2 attr-dict `:` type($rs1)";
}

#endif // GEMMINI_BALL_OPS
