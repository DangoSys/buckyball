#ifndef FIXED_OPS
#define FIXED_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"
def FixedBall_Dialect : Dialect {
  let name = "fixed_ball";
  let cppNamespace = "::buddy::fixed_ball";
  let summary = "Signed fixed-point bank operations";
}
class FixedBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<FixedBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;
def BankFixedOp : FixedBall_Op<"bank_fixed", [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "Fixed-point operations on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows,
                       I64Attr:$mode, I64Attr:$fraction, UnitAttr:$broadcast, UnitAttr:$stream);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}
def FixedOp : FixedBall_Op<"fixed", [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "Fixed-point operations on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$rows,
                       I64Attr:$mode, I64Attr:$fraction, UnitAttr:$broadcast, UnitAttr:$stream);
  let assemblyFormat = "prop-dict attr-dict";
}
#endif
