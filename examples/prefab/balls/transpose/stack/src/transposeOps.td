//===- transposeOps.td - TransposeBall operations ----------*- tablegen -*-===//
//
// transpose_i8: both banks have cols / (bankWidthBits / 8) columns. The
// instruction packs the transposed bytes into the same bank geometry, with
// `rows` bank rows.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSPOSE_OPS
#define TRANSPOSE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def TransposeBall_Dialect : Dialect {
  let name = "transpose_ball";
  let cppNamespace = "::buddy::transpose_ball";
  let summary = "The operations of TransposeBall";
}

class TransposeBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<TransposeBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankTransposeI8Op : TransposeBall_Op<"bank_transpose_i8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "transpose_i8 on virtual banks";
  let arguments = (ins I64:$input, I64:$output, I64Attr:$rows, I64Attr:$cols);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `->` $output prop-dict attr-dict";
}

def TransposeI8Op : TransposeBall_Op<"transpose_i8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "transpose_i8 on physical banks";
  let arguments = (ins I64Attr:$input, I64Attr:$output, I64Attr:$rows,
                       I64Attr:$cols);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // TRANSPOSE_OPS
