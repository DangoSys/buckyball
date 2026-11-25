//===- softmaxOps.td - SoftmaxBall operations --------------*- tablegen -*-===//
//
// softmax (input) -> output: the softmax of `rows` FP32 rows of length `w`
// (16 <= w <= 2048; ceil(w/4) bank rows each) scaled by 2^scale_log2, one SOFTMAX
// instruction (workloads/isa/softmax.h; the reference is emu/src/softmax.rs).
// Input and output are distinct banks of one 128-bit wide group each and
// start at bank row 0: the bank-row fields of the instruction are 0.
//
//===----------------------------------------------------------------------===//

#ifndef SOFTMAX_OPS
#define SOFTMAX_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def SoftmaxBall_Dialect : Dialect {
  let name = "softmax_ball";
  let cppNamespace = "::buddy::softmax_ball";
  let summary = "The operations of SoftmaxBall";
}

class SoftmaxBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<SoftmaxBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankSoftmaxOp : SoftmaxBall_Op<"bank_softmax",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "softmax on virtual banks";
  let arguments = (ins I64:$input, I64:$output, I64Attr:$rows, I64Attr:$w,
                       I64Attr:$scale_log2);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `->` $output prop-dict attr-dict";
}

def SoftmaxOp : SoftmaxBall_Op<"softmax",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "softmax on physical banks";
  let arguments = (ins I64Attr:$input, I64Attr:$output, I64Attr:$rows,
                   I64Attr:$w, I64Attr:$scale_log2);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // SOFTMAX_OPS
