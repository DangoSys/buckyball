//===-- QuantBall.td - Fp2IntBall operations ----------------*- tablegen -*-===//
//
// Fp2IntBall operations for quantization.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTBALL_OPS
#define QUANTBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

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

def BankFp2IntOp : Buckyball_Op<"bank_fp2int"> {
  let summary = "Quantization using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$scale);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $scale attr-dict `:` type($inBank) type($outBank) type($iter) type($scale)";
}

def Fp2IntIntrOp : Buckyball_IntrOpBase<"fp2int">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // QUANTBALL_OPS
