//===- smatmulOps.td - SMatMulBall operations --------------*- tablegen -*-===//
//
// matmul_i8 (a, b) -> c: c = bias + a @ b for `rows` rows (one, or a multiple
// of tileRows up to a quarter of the bank depth) and sixteen columns; it always
// reads the bias in the old c. A and B have the same number of groups; group g
// holds reduction block g of `k` columns (a multiple of sixteen), so
// K = k * groups. In group g, B holds k rows of sixteen i8. A holds k / 16 rows
// of sixteen i8 for rows = 1; otherwise M tiles of tileRows rows, each tile the
// k / 16 slices of sixteen columns one after another: line
// ((r / tileRows) * (k / 16) + c / 16) * tileRows + r % tileRows.
// C starts with the bias, one row of sixteen i32 in its first four lines, and
// receives `rows` rows of sixteen i32 (four lines each).
//
//===----------------------------------------------------------------------===//

#ifndef SMATMUL_OPS
#define SMATMUL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def SMatMulBall_Dialect : Dialect {
  let name = "smatmul_ball";
  let cppNamespace = "::buddy::smatmul_ball";
  let summary = "The operations of SMatMulBall";
}

class SMatMulBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<SMatMulBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankMatmulI8Op : SMatMulBall_Op<"bank_matmul_i8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_i8 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows, I64Attr:$k);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def MatmulI8Op : SMatMulBall_Op<"matmul_i8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_i8 on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$rows,
                       I64Attr:$k);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // SMATMUL_OPS
