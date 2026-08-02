//===-- MatrixBall.td - MatrixBall ops -------*- tablegen -*-===//

#ifndef MATRIXARRAYBALL_OPS
#define MATRIXARRAYBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

def MatrixMatmulOp : Buckyball_Op<"matrix_matmul"> {
  let summary = "Logical matmul lowered to MatrixBall (funct7=65)";
  let description = [{
    Performs C[M,N] = A[M,K] x B[K,N] with int8 inputs / int32 output.
    Memrefs must already match the MatrixBall bank tile layout for the
    given shape (single 16x16 tile for the default tile.matmul path).
    `ws = true` selects weight-stationary mode.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       MemRefRankOf<[AnyType], [2]>:$cMemArray,
                       DefaultValuedAttr<BoolAttr, "false">:$ws);
  let assemblyFormat = [{$aMemArray $bMemArray $cMemArray attr-dict `:`
    type($aMemArray) type($bMemArray) type($cMemArray)}];
}

def MatrixOp : Buckyball_Op<"matrix"> {
  let summary = "MatrixBall compute (funct7=65)";
  let description = [{
    ISA: bb_matrix_mnk_mode(op1, op2, wr, m, n, k, mode).
    `config` is the full rs2 encoding (M/N/K/mode).
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$config);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
                       "$config attr-dict `:` "
                       "type($op1BankId)";
}

def BankMatrixOp : Buckyball_Op<"bank_matrix"> {
  let summary = "MatrixBall compute with virtual bank handles";
  let arguments = (ins I64:$op1Bank, I64:$op2Bank, I64:$wrBank, I64:$config);
  let results = (outs I64:$wrBankOut);
  let assemblyFormat =
      "$op1Bank $op2Bank $wrBank $config attr-dict `:` "
      "type($op1Bank) type($op2Bank) type($wrBank) type($config)";
}

#endif // MATRIXARRAYBALL_OPS
