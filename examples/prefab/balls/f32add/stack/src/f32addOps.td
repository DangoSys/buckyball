//===- f32addOps.td - F32AddBall operations ----------------*- tablegen -*-===//
//
// add_f32: C = A + B over `rows` rows of four f32, RNE with canonical NaN.
// A, B and C are distinct one-group banks.
//
// The Ball reads both addends from its source banks and cannot read the old
// destination: a sum of more terms is a chain of add_f32 into fresh banks. The
// Ball's input group is group 0 and it always reads B.
//
//===----------------------------------------------------------------------===//

#ifndef F32ADD_OPS
#define F32ADD_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def F32AddBall_Dialect : Dialect {
  let name = "f32add_ball";
  let cppNamespace = "::buddy::f32add_ball";
  let summary = "The operations of F32AddBall";
}

class F32AddBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<F32AddBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankAddF32Op : F32AddBall_Op<"bank_add_f32",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "add_f32 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def AddF32Op : F32AddBall_Op<"add_f32",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "add_f32 on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$rows);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // F32ADD_OPS
