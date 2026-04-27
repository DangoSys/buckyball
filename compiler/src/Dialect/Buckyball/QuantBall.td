//===-- QuantBall.td - QuantBall operations ----------------*- tablegen -*-===//
//
// QuantBall operations for quantization.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTBALL_OPS
#define QUANTBALL_OPS

include "BuckyballBase.td"

//===----------------------------------------------------------------------===//
// QuantBall operations
//===----------------------------------------------------------------------===//

def QuantOp : Buckyball_Op<"quant"> {
  let summary = "Quantize floating-point to fixed-point";
  let description = [{
    Quantizes floating-point values to fixed-point representation.
    Corresponds to bb_quant instruction (funct7=51).
    Belongs to QuantBall (bid=5).
    ISA: bb_quant(bank_id, wr_bank_id, iter, scale_fp32)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$scale);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $scale attr-dict `:` "
                       "type($inputBankId)";
}

#endif // QUANTBALL_OPS
