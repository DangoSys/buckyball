#ifndef INTNORM_OPS
#define INTNORM_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def IntnormBall_Dialect : Dialect {
  let name = "intnorm_ball";
  let cppNamespace = "::buddy::intnorm_ball";
  let summary = "Integer normalization and MXINT8 packing";
}
class IntnormBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<IntnormBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankNormMxint8Op : IntnormBall_Op<"bank_norm_mxint8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let arguments = (ins I64:$input, I64:$coefficients, I64:$output,
      I64Attr:$rows, I64Attr:$k, I64Attr:$in_base,
      I64Attr:$coefficient_base, I64Attr:$out_base);
  let results = (outs I64:$outputBank);
  let assemblyFormat = "$input `,` $coefficients `->` $output prop-dict attr-dict";
}
def NormMxint8Op : IntnormBall_Op<"norm_mxint8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let arguments = (ins I64Attr:$input, I64Attr:$coefficients, I64Attr:$output,
      I64Attr:$rows, I64Attr:$k, I64Attr:$in_base,
      I64Attr:$coefficient_base, I64Attr:$out_base);
  let assemblyFormat = "prop-dict attr-dict";
}
#endif
