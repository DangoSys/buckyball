//===- maxpoolOps.td - MaxPoolBall operations --------------*- tablegen -*-===//
//
// maxpool_i8: the input bank holds a square feature map of `input_side`
// pixels per side; group g is channel block g, line y * input_side + x holds
// the sixteen i8 channels of pixel (y, x). The output, with the same groups,
// receives the `kernel` x `kernel` maxima with `stride` and `padding` in the
// same layout, output_side = (input_side + 2 * padding - kernel) / stride + 1.
//
//===----------------------------------------------------------------------===//

#ifndef MAXPOOL_OPS
#define MAXPOOL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def MaxPoolBall_Dialect : Dialect {
  let name = "maxpool_ball";
  let cppNamespace = "::buddy::maxpool_ball";
  let summary = "The operations of MaxPoolBall";
}

class MaxPoolBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<MaxPoolBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankMaxpoolI8Op : MaxPoolBall_Op<"bank_maxpool_i8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "maxpool_i8 on virtual banks";
  let arguments = (ins I64:$input, I64:$output, I64Attr:$input_side,
                       I64Attr:$kernel, I64Attr:$stride, I64Attr:$padding);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `->` $output prop-dict attr-dict";
}

def MaxpoolI8Op : MaxPoolBall_Op<"maxpool_i8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "maxpool_i8 on physical banks";
  let arguments = (ins I64Attr:$input, I64Attr:$output, I64Attr:$input_side,
                       I64Attr:$kernel, I64Attr:$stride, I64Attr:$padding);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MAXPOOL_OPS
