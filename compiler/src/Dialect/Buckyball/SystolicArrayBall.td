//===-- SystolicArrayBall.td - SystolicArrayBall ops -------*- tablegen -*-===//
//
// SystolicArrayBall operations for systolic array computation.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTOLICARRAYBALL_OPS
#define SYSTOLICARRAYBALL_OPS

include "BuckyballBase.td"

//===----------------------------------------------------------------------===//
// SystolicArrayBall operations
//===----------------------------------------------------------------------===//

def SystolicOp : Buckyball_Op<"systolic"> {
  let summary = "Systolic array computation (BBFP multiply)";
  let description = [{
    Performs computation using systolic array with block floating point.
    Corresponds to bb_bbfp_mul instruction (funct7=65).
    Belongs to SystolicArrayBall (bid=4).
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$config);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` "
                       "$resultBankId `,` $config attr-dict `:` "
                       "type($op1BankId)";
}

#endif // SYSTOLICARRAYBALL_OPS
