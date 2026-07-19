//===-- VecBall.td - VecBall operations --------------------*- tablegen -*-===//
//
// VecBall operations for matrix multiplication using warp-16 matrix array.
//
//===----------------------------------------------------------------------===//

#ifndef VECBALL_OPS
#define VECBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

//===----------------------------------------------------------------------===//
// VecBall operations
//===----------------------------------------------------------------------===//

def MulWarp16Op : Buckyball_Op<"mul_warp16"> {
  let summary = "Warp-16 matrix multiplication";
  let description = [{
    Performs matrix multiplication using 16x16 matrix array.
    Corresponds to bb_mul_warp16 instruction (funct7=64).
    Belongs to VecBall (bid=0).
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$wrBankId, I64:$iter, I64:$mode);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $wrBankId `,` "
                       "$iter `,` $mode attr-dict `:` "
                       "type($op1BankId)";
}

def BankMulWarp16Op : Buckyball_Op<"bank_mul_warp16"> {
  let summary = "Warp-16 matmul using virtual bank handles";
  let arguments = (ins I64:$op1Bank, I64:$op2Bank, I64:$wrBank,
                       I64:$iter, I64:$mode);
  let results = (outs I64:$wrBankOut);
  let assemblyFormat =
      "$op1Bank $op2Bank $wrBank $iter $mode attr-dict `:` type($op1Bank) type($op2Bank) type($wrBank) type($iter) type($mode)";
}

def MulWarp16IntrOp : Buckyball_IntrOpBase<"mul.warp16">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // VECBALL_OPS
