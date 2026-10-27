//===- reluOps.td - ReluBall operations --------------------*- tablegen -*-===//
//
// relu_i32: max(x, 0) on the first `rows` rows of four i32 of a one-group
// bank, in place: the one source is the destination.
//
// The Ball updates whole groups of sixteen rows: it rounds `rows` up to a
// multiple of sixteen, which only touches rows past the valid data, and runs
// one segment as deep as the bank.
//
//===----------------------------------------------------------------------===//

#ifndef RELU_OPS
#define RELU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def ReluBall_Dialect : Dialect {
  let name = "relu_ball";
  let cppNamespace = "::buddy::relu_ball";
  let summary = "The operations of ReluBall";
}

class ReluBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<ReluBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankReluI32Op : ReluBall_Op<"bank_relu_i32",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "relu_i32 on a virtual bank, in place";
  let arguments = (ins I64:$bank, I64Attr:$rows);
  let results = (outs I64:$bankOut);
  let assemblyFormat = "$bank prop-dict attr-dict";
}

def ReluI32Op : ReluBall_Op<"relu_i32",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "relu_i32 on a physical bank, in place";
  let arguments = (ins I64Attr:$bank, I64Attr:$rows);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // RELU_OPS
