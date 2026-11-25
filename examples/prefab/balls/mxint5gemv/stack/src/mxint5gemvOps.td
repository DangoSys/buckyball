#ifndef MXINT5GEMV_OPS
#define MXINT5GEMV_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def Mxint5gemvBall_Dialect : Dialect {
  let name = "mxint5gemv_ball";
  let cppNamespace = "::buddy::mxint5gemv_ball";
  let summary = "Integer GEMV with four weight-bank groups";
}
class Mxint5gemvBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mxint5gemvBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankGemvMxint5Op : Mxint5gemvBall_Op<"bank_gemv_mxint5",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$k,
    I64Attr:$a_codes, I64Attr:$a_exponents, I64Attr:$b_base,
    I64:$c_base, BoolAttr:$accumulate, BoolAttr:$finalize, BoolAttr:$update);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c `at` $c_base prop-dict attr-dict";
}

def GemvMxint5Op : Mxint5gemvBall_Op<"gemv_mxint5",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$k,
    I64Attr:$a_codes, I64Attr:$a_exponents, I64Attr:$b_base,
    I64:$c_base, BoolAttr:$accumulate, BoolAttr:$finalize, BoolAttr:$update);
  let assemblyFormat = "$c_base prop-dict attr-dict";
}
#endif
