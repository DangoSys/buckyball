//===- mataddOps.td - MatAddBall operations ----------------*- tablegen -*-===//
//
// add_i32: C = A + B modulo 2^32 over `rows` rows of four i32 in every group
// of the banks; A, B and C have the same number of groups.
//
//===----------------------------------------------------------------------===//

#ifndef MATADD_OPS
#define MATADD_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def MatAddBall_Dialect : Dialect {
  let name = "matadd_ball";
  let cppNamespace = "::buddy::matadd_ball";
  let summary = "The operations of MatAddBall";
}

class MatAddBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<MatAddBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankAddI32Op : MatAddBall_Op<"bank_add_i32",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "add_i32 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def AddI32Op : MatAddBall_Op<"add_i32",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "add_i32 on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$rows);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MATADD_OPS
