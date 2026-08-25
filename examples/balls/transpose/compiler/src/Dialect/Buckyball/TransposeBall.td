//===-- TransposeBall.td - TransposeBall operations --------*- tablegen -*-===//
//
// TransposeBall operations for matrix transpose.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPOSEBALL_OPS
#define TRANSPOSEBALL_OPS

include "Buckyball.td"

// Memref-level transpose (like smatmul_matmul). Chip bank-ssa decides
// pad / tile / Bank* composition; this op carries no bank policy.
def MemTransposeOp : Buckyball_Op<"mem_transpose"> {
  let summary = "Logical matrix transpose on memrefs";
  let description = [{
    Transposes input[R,C] into output[C,R]. Element type and layout
    constraints are enforced by the chip bank-ssa lowering.
  }];
  let arguments = (ins MemRefRankOf<[AnyType], [2]>:$input,
                       MemRefRankOf<[AnyType], [2]>:$output);
  let assemblyFormat = [{$input $output attr-dict `:`
    type($input) type($output)}];
}

def TransposeOp : Buckyball_Op<"transpose"> {
  let summary = "Matrix transpose operation";
  let description = [{
    Transposes iter x W to W x iter in scratchpad banks.
    ISA: bb_transpose(op1_bank, wr_bank, iter, elem_bits); funct7 from Core ballISA.
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
