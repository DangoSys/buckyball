#ifndef MXINT8UNPACK_OPS
#define MXINT8UNPACK_OPS
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def Mxint8UnpackBall_Dialect : Dialect {
  let name = "mxint8unpack_ball";
  let cppNamespace = "::buddy::mxint8unpack_ball";
  let summary = "Packed MXINT8 to exact IEEE32 bit expansion";
}
class Mxint8UnpackBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mxint8UnpackBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankUnpackOp : Mxint8UnpackBall_Op<"bank_unpack",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let arguments = (ins I64:$input, I64:$output, I64:$blocks);
  let results = (outs I64:$outputState);
  let assemblyFormat = "$input `->` $output `blocks` $blocks attr-dict";
}

def UnpackOp : Mxint8UnpackBall_Op<"unpack",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let arguments = (ins I64Attr:$input, I64Attr:$output, I64:$blocks);
  let assemblyFormat = "$blocks prop-dict attr-dict";
}
#endif
