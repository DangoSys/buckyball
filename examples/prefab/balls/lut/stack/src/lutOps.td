//===- lutOps.td - LutBall operations ----------------------*- tablegen -*-===//
//
// lookup_i8: `codes` holds `rows` rows of sixteen i8 codes, `table` the
// table: one group of 256 entries shared by every channel, or four groups with
// 256 entries per channel (channel * 256 + code, 1024 bytes per group). The
// output receives table[code] for every element.
//
//===----------------------------------------------------------------------===//

#ifndef LUT_OPS
#define LUT_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def LutBall_Dialect : Dialect {
  let name = "lut_ball";
  let cppNamespace = "::buddy::lut_ball";
  let summary = "The operations of LutBall";
}

class LutBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<LutBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankLookupI8Op : LutBall_Op<"bank_lookup_i8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "lookup_i8 on virtual banks";
  let arguments = (ins I64:$codes, I64:$table, I64:$output, I64Attr:$rows);
  let results = (outs I64:$outputOut);
  let assemblyFormat = "$codes `,` $table `->` $output prop-dict attr-dict";
}

def LookupI8Op : LutBall_Op<"lookup_i8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "lookup_i8 on physical banks";
  let arguments = (ins I64Attr:$codes, I64Attr:$table, I64Attr:$output,
                       I64Attr:$rows);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // LUT_OPS
