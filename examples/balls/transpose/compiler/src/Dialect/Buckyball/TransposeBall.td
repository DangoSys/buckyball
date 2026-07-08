//===-- TransposeBall.td - TransposeBall operations --------*- tablegen -*-===//
//
// TransposeBall operations for matrix transpose.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPOSEBALL_OPS
#define TRANSPOSEBALL_OPS

include "BuckyballBase.td"
include "BuckyballIntrinsics.td"

//===----------------------------------------------------------------------===//
// TransposeBall operations
//===----------------------------------------------------------------------===//

def TransposeOp : Buckyball_Op<"transpose"> {
  let summary = "Matrix transpose operation";
  let description = [{
    Transposes a matrix in scratchpad memory.
    Corresponds to bb_transpose instruction (funct7=49).
    Belongs to TransposeBall (bid=2).
    ISA: bb_transpose(op1_bank_id, wr_bank_id, iter, mode)
  }];
  let arguments = (ins I64:$inputBankId, I64:$outputBankId,
                       I64:$iter, I64:$mode);
  let assemblyFormat = "$inputBankId `,` $outputBankId `,` "
                       "$iter `,` $mode attr-dict `:` "
                       "type($inputBankId)";
}

def BankTransposeOp : Buckyball_Op<"bank_transpose"> {
  let summary = "Transpose using virtual bank handles";
  let arguments = (ins I64:$inBank, I64:$outBank, I64:$iter, I64:$mode);
  let results = (outs I64:$outBankOut);
  let assemblyFormat =
      "$inBank $outBank $iter $mode attr-dict `:` type($inBank) type($outBank) type($iter) type($mode)";
}

def TransposeIntrOp : Buckyball_IntrOpBase<"transpose">,
  Arguments<(ins LLVM_Type, LLVM_Type)>;

#endif // TRANSPOSEBALL_OPS
