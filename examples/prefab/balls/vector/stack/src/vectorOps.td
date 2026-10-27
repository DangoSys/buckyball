//===- vectorOps.td - VecBall operations -------------------*- tablegen -*-===//
//
// matmul_at_i8 (at, b) -> c: c += at^T * b, one VECMAT16 instruction; it
// always reads the old c. Banks are one group of sixteen i8 per row unless
// noted:
//   at: A^T, `rows` rows; row k is column k of the 16 x rows matrix A.
//   b: `rows` rows of sixteen i8.
//   c: a bank of four groups holding sixteen rows of sixteen i32; group g
//     holds columns 4g..4g+3 of each row, so C moves to and from memory as a
//     plain 16 x 16 i32 matrix.
// `rows` is the reduction depth K, 1..bank depth.
//
//===----------------------------------------------------------------------===//

#ifndef VECTOR_OPS
#define VECTOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def VecBall_Dialect : Dialect {
  let name = "vector_ball";
  let cppNamespace = "::buddy::vector_ball";
  let summary = "The operations of VecBall";
}

class VecBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<VecBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankMatmulAtI8Op : VecBall_Op<"bank_matmul_at_i8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_at_i8 on virtual banks";
  let arguments = (ins I64:$at, I64:$b, I64:$c, I64Attr:$rows);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$at `,` $b `->` $c prop-dict attr-dict";
}

def MatmulAtI8Op : VecBall_Op<"matmul_at_i8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_at_i8 on physical banks";
  let arguments = (ins I64Attr:$at, I64Attr:$b, I64Attr:$c, I64Attr:$rows);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // VECTOR_OPS
