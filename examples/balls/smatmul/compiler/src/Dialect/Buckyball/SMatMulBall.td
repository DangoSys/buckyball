//===-- SMatMulBall.td - SMatMulBall ops -------*- tablegen -*-===//

#ifndef SMATMULBALL_OPS
#define SMATMULBALL_OPS

include "Buckyball.td"

def SMatMulMatmulOp : Buckyball_Op<"smatmul_matmul"> {
  let summary = "Logical matmul lowered to SMatMulBall";
  let description = [{
    Performs C[M,N] = A[M,K] x B[K,N] with int8 inputs / int32 output.
    Memrefs must already match the SMatMulBall bank tile layout. Logical N
    may exceed 16; bank lowering emits one 16-lane C panel at a time.
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
  let summary = "SMatMulBall compute";
  let description = [{
    ISA: SMATMUL_OS. `config` contains rows, cols, and k.
  }];
  let arguments = (ins I64:$op1BankId, I64:$op2BankId,
                       I64:$resultBankId, I64:$config,
                       I1:$first, I1:$last, I64:$outputBase);
  let assemblyFormat = "$op1BankId `,` $op2BankId `,` $resultBankId `,` "
                       "$config `,` $first `,` $last `,` $outputBase "
                       "attr-dict `:` "
                       "type($op1BankId)";
}

def SMatMulBiasOp : Buckyball_Op<"smatmul_bias"> {
  let summary = "Preload one 16-channel INT32 bias vector";
  let description = [{
    Reads four 128-bit SRAM rows containing 16 INT32 bias values. SMatMulBall
    uses the retained vector to initialize every new 16x16 accumulator tile.
  }];
  let arguments = (ins I64:$biasBankId, I64:$inputBase);
  let assemblyFormat = "$biasBankId `,` $inputBase attr-dict `:` "
                       "type($biasBankId)";
}

def BankSMatMulOp : Buckyball_Op<"bank_smatmul"> {
  let summary = "SMatMulBall compute with virtual bank handles";
  let arguments = (ins I64:$op1Bank, I64:$op2Bank, I64:$wrBank, I64:$config,
                       I1:$first, I1:$last, I64:$outputBase);
  let results = (outs I64:$wrBankOut);
  let assemblyFormat =
      "$op1Bank $op2Bank $wrBank $config $first $last $outputBase "
      "attr-dict `:` "
      "type($op1Bank) type($op2Bank) type($wrBank) type($config)";
}

def BankSMatMulBiasOp : Buckyball_Op<"bank_smatmul_bias"> {
  let summary = "SMatMul bias preload with a virtual bank handle";
  let arguments = (ins I64:$biasBank, I64:$inputBase);
  let results = (outs I64:$biasBankOut);
  let assemblyFormat = "$biasBank $inputBase attr-dict `:` "
                       "type($biasBank) type($inputBase)";
}

#endif // SMATMULBALL_OPS
