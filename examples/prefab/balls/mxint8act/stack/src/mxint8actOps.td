//===- mxint8actOps.td - Mxint8actBall operations ----------*- tablegen -*-===//
//
// act_mxint8: `input` holds `rows` packed MXINT8 rows of `k` codes (followed
// by `rows` gate rows when gated), `table` the 4096-entry Q26 activation table
// (1024 bank rows). The output receives the activated (and gated) rows,
// requantized per block of 32 values. CONTRACT.md has the definition.
//
//===----------------------------------------------------------------------===//

#ifndef MXINT8ACT_OPS
#define MXINT8ACT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def Mxint8actBall_Dialect : Dialect {
  let name = "mxint8act_ball";
  let cppNamespace = "::buddy::mxint8act_ball";
  let summary = "The operations of Mxint8actBall";
}

class Mxint8actBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mxint8actBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankActMxint8Op : Mxint8actBall_Op<"bank_act_mxint8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "Integer activation and optional gate on packed MXINT8 banks";
  let arguments = (ins I64:$input, I64:$table, I64:$output, I64Attr:$rows,
      I64Attr:$k, I64Attr:$in_base, I64Attr:$out_base, I64Attr:$table_base, BoolAttr:$gated);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$input `,` $table `->` $output prop-dict attr-dict";
}

def ActMxint8Op : Mxint8actBall_Op<"act_mxint8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "Integer activation on physical MXINT8 banks";
  let arguments = (ins I64Attr:$input, I64Attr:$table, I64Attr:$output, I64Attr:$rows,
      I64Attr:$k, I64Attr:$in_base, I64Attr:$out_base, I64Attr:$table_base, BoolAttr:$gated);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MXINT8ACT_OPS
