//===-- SMatMulBall.td - SMatMulBall ops -------*- tablegen -*-===//

#ifndef SMATMULBALL_OPS
#define SMATMULBALL_OPS

include "Buckyball.td"

def SMatMulMatmulOp : Buckyball_Op<"smatmul_matmul"> {
  let summary = "Logical matmul lowered to SMatMulBall (OS / WS)";
  let description = [{
    Performs C[M,N] = A[M,K] x B[K,N] with int8 inputs / int32 output.
    Memrefs must already match the SMatMulBall bank tile layout. Logical N
    may exceed 16; bank lowering emits one 16-lane C panel at a time.
    Compiler emits bb_smatmul_ws when ceil(M/16) >= 2, else bb_smatmul_os.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$aMemArray,
                       MemRefRankOf<[AnyType], [2]>:$bMemArray,
                       MemRefRankOf<[AnyType], [2]>:$cMemArray,
                       DefaultValuedAttr<I64Attr, "-1">:$dwAddr,
                       DefaultValuedAttr<I64Attr, "0">:$dwBytes,
                       DefaultValuedAttr<BoolAttr, "false">:$perChannel);
  let assemblyFormat = [{$aMemArray $bMemArray $cMemArray attr-dict `:`
    type($aMemArray) type($bMemArray) type($cMemArray)}];
}

def SMatMulOp : Buckyball_Op<"smatmul"> {
  let summary = "SMatMulBall compute (OS / WS)";
  let description = [{
    ISA: bb_smatmul_os / bb_smatmul_ws(op1, op2, wr, rows, cols, k).
    `config` is rs2 (rows/cols/k). cols must be 1..16.
    `ws` selects the dataflow. The Core ballISA map assigns the instruction
    encoding (SMATMUL_OS / SMATMUL_WS).
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$config,
                       DefaultValuedAttr<I64Attr, "0">:$op1Base,
                       DefaultValuedAttr<I64Attr, "0">:$op2Base,
                       DefaultValuedAttr<I64Attr, "0">:$wrBase,
                       DefaultValuedAttr<BoolAttr, "false">:$ws);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
                       "$config attr-dict `:` "
                       "type($op1BankId)";
}

def BankSMatMulOp : Buckyball_Op<"bank_smatmul"> {
  let summary = "SMatMulBall compute with virtual bank handles";
  let arguments = (ins I64:$op1Bank, I64:$op2Bank, I64:$wrBank, I64:$config,
                       DefaultValuedAttr<I64Attr, "0">:$op1Base,
                       DefaultValuedAttr<I64Attr, "0">:$op2Base,
                       DefaultValuedAttr<I64Attr, "0">:$wrBase,
                       DefaultValuedAttr<BoolAttr, "false">:$ws);
  let results = (outs I64:$wrBankOut);
  let assemblyFormat =
      "$op1Bank $op2Bank $wrBank $config attr-dict `:` "
      "type($op1Bank) type($op2Bank) type($wrBank) type($config)";
}

#endif // SMATMULBALL_OPS
