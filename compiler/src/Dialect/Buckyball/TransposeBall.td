//===-- TransposeBall.td - TransposeBall operations --------*- tablegen -*-===//
//
// TransposeBall operations for matrix transpose.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPOSEBALL_OPS
#define TRANSPOSEBALL_OPS

include "BuckyballBase.td"

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

#endif // TRANSPOSEBALL_OPS
