//===-- MatrixBall.td - MatrixBall ops -------*- tablegen -*-===//
//
// MatrixBall operations for matrix array computation.
//
//===----------------------------------------------------------------------===//

#ifndef MATRIXARRAYBALL_OPS
#define MATRIXARRAYBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

//===----------------------------------------------------------------------===//
// MatrixBall operations
//===----------------------------------------------------------------------===//

def MatrixMatmulOp : Buckyball_Op<"matrix_matmul"> {
  let summary = "Logical NxN i8 matmul lowered to the prototype MatrixBall";
  let description = [{
    Performs a logical square i8 matmul and writes an i32 result matrix.
    Lowering packs the logical matrices into the bank-line layout expected by
    the current MatrixBall prototype. By default this lowers to the
    MatrixBall OS path. Setting `ws = true` selects the WS path.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       MemRefRankOf<[AnyType], [2]>:$cMemArray,
                       DefaultValuedAttr<BoolAttr, "false">:$ws);
  let assemblyFormat = [{$aMemArray $bMemArray $cMemArray attr-dict `:`
    type($aMemArray) type($bMemArray) type($cMemArray)}];
}

def MatrixOp : Buckyball_Op<"matrix"> {
  let summary = "Matrix array computation (matrix multiply)";
  let description = [{
    Performs computation using the matrix array.
    Corresponds to the matrix instruction (funct7=65).
    Belongs to MatrixBall (bid=4).
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$config);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
                       "$config attr-dict `:` "
                       "type($op1BankId)";
}

def MatrixIterOp : Buckyball_Op<"matrix_iter"> {
  let summary = "Matrix array computation with explicit iter";
  let description = [{
    Performs computation using matrix array with block floating point and an
    explicit iter field encoded into rs1. This op is used by the new tiled
    MatrixBall compiler path and keeps the legacy matrix op ABI
    unchanged for direct users.
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$iter, I64:$config);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
                       "$iter `,` $config attr-dict `:` "
                       "type($op1BankId)";
}

def MatrixIntrOp : Buckyball_IntrOpBase<"bbfp.mul">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // MATRIXARRAYBALL_OPS
