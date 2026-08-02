//===-- TransposeBall.td - TransposeBall operations --------*- tablegen -*-===//
//
// TransposeBall operations for matrix transpose.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPOSEBALL_OPS
#define TRANSPOSEBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

def TransposeOp : Buckyball_Op<"transpose"> {
  let summary = "Matrix transpose operation";
  let description = [{
    Transposes iter x W to W x iter in scratchpad banks.
    funct7=49. ISA: bb_transpose(op1_bank, wr_bank, iter, elem_bits)
    W = (bankWidth * cols) / elem_bits.
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$elemBits);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $elemBits attr-dict `:` "
                       "type($inputBankId)";
}

def BankTransposeOp : Buckyball_Op<"bank_transpose"> {
  let summary = "Transpose using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$elemBits);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $elemBits attr-dict `:` type($inBank) type($outBank) type($iter) type($elemBits)";
}

def TransposeIntrOp : Buckyball_IntrOpBase<"transpose">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // TRANSPOSEBALL_OPS
