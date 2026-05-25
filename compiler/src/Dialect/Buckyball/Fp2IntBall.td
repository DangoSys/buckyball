//===-- QuantBall.td - Fp2IntBall operations ----------------*- tablegen -*-===//
//
// Fp2IntBall operations for quantization.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTBALL_OPS
#define QUANTBALL_OPS

include "BuckyballBase.td"

//===----------------------------------------------------------------------===//
// Fp2IntBall operations
//===----------------------------------------------------------------------===//

def Fp2IntOp : Buckyball_Op<"fp2int"> {
  let summary = "Quantize floating-point to fixed-point";
  let description = [{
    Quantizes floating-point values to fixed-point representation.
    Corresponds to bb_fp2int instruction (funct7=51).
    Belongs to Fp2IntBall (bid=5).
    ISA: bb_fp2int(bank_id, wr_bank_id, iter, scale_fp32)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$scale);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $scale attr-dict `:` "
                       "type($inputBankId)";
}

#endif // QUANTBALL_OPS
