//===-- DequantBall.td - DequantBall operations ------------*- tablegen -*-===//
//
// DequantBall operations for dequantization.
//
//===----------------------------------------------------------------------===//

#ifndef DEQUANTBALL_OPS
#define DEQUANTBALL_OPS

include "BuckyballBase.td"

//===----------------------------------------------------------------------===//
// DequantBall operations
//===----------------------------------------------------------------------===//

def DequantOp : Buckyball_Op<"dequant"> {
  let summary = "Dequantize fixed-point to floating-point";
  let description = [{
    Dequantizes fixed-point values back to floating-point representation.
    Corresponds to bb_dequant instruction (funct7=52).
    Belongs to DequantBall (bid=6).
    ISA: bb_dequant(bank_id, wr_bank_id, iter, scale_fp32)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$scale);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $scale attr-dict `:` "
                       "type($inputBankId)";
}

#endif // DEQUANTBALL_OPS
