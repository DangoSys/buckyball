//===- redmuleOps.td - RedmuleBall operations --------------*- tablegen -*-===//
//
// matmul_f16: C = A @ B in FP16 with ordered FMAs, or C = C + A @ B with
// read_destination. `k` is a positive multiple of eight.
//   A: `rows` rows of `k` f16, k / 8 bank rows each.
//   B: `k` bank rows of eight f16.
//   C: `rows` bank rows of eight f16.
//
// The Ball places every operand at bank row 0. With read_destination rs1 marks
// the old C, which the Ball reads from the bank the destination held before
// this write, and the FMAs start from it instead of positive zero.
//
//===----------------------------------------------------------------------===//

#ifndef REDMULE_OPS
#define REDMULE_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def RedmuleBall_Dialect : Dialect {
  let name = "redmule_ball";
  let cppNamespace = "::buddy::redmule_ball";
  let summary = "The operations of RedmuleBall";
}

class RedmuleBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<RedmuleBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankMatmulF16Op : RedmuleBall_Op<"bank_matmul_f16",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_f16 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows, I64Attr:$k,
                       UnitAttr:$read_destination);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def MatmulF16Op : RedmuleBall_Op<"matmul_f16",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_f16 on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$rows,
                       I64Attr:$k, UnitAttr:$read_destination);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // REDMULE_OPS
