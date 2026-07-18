//===-- SystolicArrayBall.td - SystolicArrayBall ops -------*- tablegen -*-===//
//
// SystolicArrayBall operations for systolic array computation.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTOLICARRAYBALL_OPS
#define SYSTOLICARRAYBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

//===----------------------------------------------------------------------===//
// SystolicArrayBall operations
//===----------------------------------------------------------------------===//

def SystolicMatmulOp : Buckyball_Op<"systolic_matmul"> {
  let summary = "Logical NxN i8 matmul lowered to the prototype SystolicArrayBall";
  let description = [{
    Performs a logical square i8 matmul and writes an i32 result matrix.
    Lowering packs the logical matrices into the bank-line layout expected by
    the current SystolicArrayBall prototype. By default this lowers to the
    SystolicArrayBall OS path. Setting `ws = true` selects the WS path.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       MemRefRankOf<[AnyType], [2]>:$cMemArray,
                       DefaultValuedAttr<BoolAttr, "false">:$ws);
  let assemblyFormat = [{$aMemArray $bMemArray $cMemArray attr-dict `:`
    type($aMemArray) type($bMemArray) type($cMemArray)}];
}

def SystolicOp : Buckyball_Op<"systolic"> {
  let summary = "Systolic array computation (BBFP multiply)";
  let description = [{
    Performs computation using systolic array with block floating point.
    Corresponds to bb_bbfp_mul instruction (funct7=65).
    Belongs to SystolicArrayBall (bid=4).
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$config);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
                       "$config attr-dict `:` "
                       "type($op1BankId)";
}

def SystolicIterOp : Buckyball_Op<"systolic_iter"> {
  let summary = "Systolic array computation with explicit iter";
  let description = [{
    Performs computation using systolic array with block floating point and an
    explicit iter field encoded into rs1. This op is used by the new tiled
    SystolicArrayBall compiler path and keeps the legacy systolic op ABI
    unchanged for direct users.
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$iter, I64:$config);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
                       "$iter `,` $config attr-dict `:` "
                       "type($op1BankId)";
}

def SystolicIntrOp : Buckyball_IntrOpBase<"bbfp.mul">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // SYSTOLICARRAYBALL_OPS
