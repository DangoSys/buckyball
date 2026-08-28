//===-- VecBall.td - VecBall operations --------------------*- tablegen -*-===//
//
// VecBall operations for matrix multiplication using warp-16 matrix array.
//
//===----------------------------------------------------------------------===//

#ifndef VECBALL_OPS
#define VECBALL_OPS

include "Buckyball.td"

//===----------------------------------------------------------------------===//
// VecBall operations
//===----------------------------------------------------------------------===//

def VecMat16Op : Buckyball_Op<"vecmat16"> {
  let summary = "Warp-16 matrix multiplication";
  let description = [{
    Performs matrix multiplication using 16x16 matrix array.
    Corresponds to bb_vecmat16 instruction (funct7 from Core ballISA).
    Belongs to VecBall (bid=0).
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$wrBankId, I64:$iter, I64:$mode);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $wrBankId `,` "
                       "$iter `,` $mode attr-dict `:` "
                       "type($op1BankId)";
}

def BankVecMat16Op : Buckyball_Op<"bank_vecmat16"> {
  let summary = "Warp-16 matmul using virtual bank handles";
  let arguments = (ins I64:$op1Bank, I64:$op2Bank, I64:$wrBank,
                       I64:$iter, I64:$mode);
  let results = (outs I64:$wrBankOut);
  let assemblyFormat =
      "$op1Bank $op2Bank $wrBank $iter $mode attr-dict `:` type($op1Bank) type($op2Bank) type($wrBank) type($iter) type($mode)";
}

#endif // VECBALL_OPS
