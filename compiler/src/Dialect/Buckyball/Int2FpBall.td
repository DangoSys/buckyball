//===-- DequantBall.td - Int2FpBall operations ------------*- tablegen -*-===//
//
// Int2FpBall operations for dequantization.
//
//===----------------------------------------------------------------------===//

#ifndef DEQUANTBALL_OPS
#define DEQUANTBALL_OPS

include "BuckyballBase.td"

//===----------------------------------------------------------------------===//
// Int2FpBall operations
//===----------------------------------------------------------------------===//

def Int2FpOp : Buckyball_Op<"int2fp"> {
  let summary = "Dequantize fixed-point to floating-point";
  let description = [{
    Dequantizes fixed-point values back to floating-point representation.
    Corresponds to bb_int2fp instruction (funct7=52).
    Belongs to Int2FpBall (bid=6).
    ISA: bb_int2fp(bank_id, wr_bank_id, iter, scale_fp32)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$scale);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $scale attr-dict `:` "
                       "type($inputBankId)";
}

#endif // DEQUANTBALL_OPS
