//===-- Int2FpBall.td - Int2FpBall operations --------------*- tablegen -*-===//
//
// Int2FpBall operations for dequantization / requantization.
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
  let summary = "Dequantize fixed-point to floating-point";
  let description = [{
    Dequantizes fixed-point values back to floating-point representation.
    Corresponds to bb_int2fp instruction (funct7=52, output_mode=0).
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
  let summary = "Dequantization using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$scale);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $scale attr-dict `:` type($inBank) type($outBank) type($iter) type($scale)";
}

def Int32ToInt8Op : Buckyball_Op<"int32_to_int8"> {
  let summary = "Requantize INT32 bank groups to packed INT8";
  let description = [{
    Requantizes four INT32 bank groups into one packed INT8 bank.
    Corresponds to bb_int32_to_int8 (funct7=52, output_mode=1).
    Belongs to Int2FpBall (ballName=Int2FpBall; ballId is chip-config specific).
    ISA: bb_int32_to_int8(bank_id, wr_bank_id, iter, scale_fp32)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$scale);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $scale attr-dict `:` "
                       "type($inputBankId)";
}

def BankInt32ToInt8Op : Buckyball_Op<"bank_int32_to_int8"> {
  let summary = "INT32 to INT8 requant using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$scale);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $scale attr-dict `:` type($inBank) type($outBank) type($iter) type($scale)";
}

def Int2FpIntrOp : Buckyball_IntrOpBase<"int2fp">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // DEQUANTBALL_OPS
