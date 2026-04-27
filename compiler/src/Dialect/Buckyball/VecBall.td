//===-- VecBall.td - VecBall operations --------------------*- tablegen -*-===//
//
// VecBall operations for matrix multiplication using warp-16 systolic array.
//
//===----------------------------------------------------------------------===//

#ifndef VECBALL_OPS
#define VECBALL_OPS

include "BuckyballBase.td"

//===----------------------------------------------------------------------===//
// VecBall operations
//===----------------------------------------------------------------------===//

def MulWarp16Op : Buckyball_Op<"mul_warp16"> {
  let summary = "Warp-16 matrix multiplication";
  let description = [{
    Performs matrix multiplication using 16x16 systolic array.
    Corresponds to bb_mul_warp16 instruction (funct7=64).
    Belongs to VecBall (bid=0).
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$wrBankId, I64:$iter, I64:$mode);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $wrBankId `,` "
                       "$iter `,` $mode attr-dict `:` "
                       "type($op1BankId)";
}

#endif // VECBALL_OPS
