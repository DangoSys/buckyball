//===-- Int2FpBall.td - Int2FpBall operations --------------*- tablegen -*-===//
//
// Int2FpBall operation for INT32-to-FP32 conversion.
//
//===----------------------------------------------------------------------===//

#ifndef DEQUANTBALL_OPS
#define DEQUANTBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

//===----------------------------------------------------------------------===//
// Int2FpBall operations
//===----------------------------------------------------------------------===//

def Int2FpOp : Buckyball_Op<"int2fp"> {
  let summary = "Convert INT32 to FP32";
  let description = [{
    Converts INT32 values to FP32 and applies the FP32 scale multiplier.
    Corresponds to bb_int2fp instruction (funct7=52).
    Belongs to Int2FpBall (ballName=Int2FpBall; ballId is chip-config specific).
    ISA: bb_int2fp(bank_id, wr_bank_id, iter, scale_fp32)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$scale);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $scale attr-dict `:` "
                       "type($inputBankId)";
}

def BankInt2FpOp : Buckyball_Op<"bank_int2fp"> {
  let summary = "INT32-to-FP32 conversion using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$scale);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $scale attr-dict `:` type($inBank) type($outBank) type($iter) type($scale)";
}

def Int2FpIntrOp : Buckyball_IntrOpBase<"int2fp">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // DEQUANTBALL_OPS
