#ifndef MXINT8GEMV_OPS
#define MXINT8GEMV_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def Mxint8gemvBall_Dialect : Dialect {
  let name = "mxint8gemv_ball";
  let cppNamespace = "::buddy::mxint8gemv_ball";
  let summary = "Integer GEMV with four weight-bank groups";
}
class Mxint8gemvBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mxint8gemvBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankGemvMxint8Op : Mxint8gemvBall_Op<"bank_gemv_mxint8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$k,
    I64Attr:$a_codes, I64Attr:$a_exponents, I64Attr:$b_base,
    I64Attr:$c_base, BoolAttr:$accumulate, BoolAttr:$finalize);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def GemvMxint8Op : Mxint8gemvBall_Op<"gemv_mxint8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$k,
    I64Attr:$a_codes, I64Attr:$a_exponents, I64Attr:$b_base,
    I64Attr:$c_base, BoolAttr:$accumulate, BoolAttr:$finalize);
  let assemblyFormat = "prop-dict attr-dict";
}
#endif
