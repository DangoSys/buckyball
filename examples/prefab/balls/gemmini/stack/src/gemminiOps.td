//===- gemminiOps.td - GemminiBall operations --------------*- tablegen -*-===//
//
// matmul_i8_tile: one 16x16 tile C = A @ B over `k`, a positive multiple of
// sixteen. A and B fill their groups in order, each group a bank depth of rows
// before the next, so a K deeper than a bank spans several groups (allocated
// together or gathered with bank_transfer).
//   A: k / 16 blocks of sixteen rows of sixteen i8; row i of block r is
//      A[i, 16r .. 16r + 15].
//   B: k rows of sixteen i8, B[0 .. k - 1, 0 .. 15].
//   C: a bank of four groups, sixteen rows of sixteen i32.
// With read_destination C = in_shift(C + A @ B, shift), rounding to nearest
// even; without, C = A @ B and there is no shift.
//
// This is not smatmul's matmul_i8: C holds one output row per bank row across
// four groups instead of four bank rows of one group, there is no bias row, and
// read_destination accumulates the whole old C with a rounding shift.
//
// The Ball issues one instruction per 16-deep block, the operand groups and row
// bases derived from the block index. Without read_destination the first block
// runs on the output-stationary array (PRELOAD primes D = 0); every other block
// accumulates on the weight-stationary array, reading the old C. Only the last
// block applies the shift.
//
//===----------------------------------------------------------------------===//

#ifndef GEMMINI_OPS
#define GEMMINI_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def GemminiBall_Dialect : Dialect {
  let name = "gemmini_ball";
  let cppNamespace = "::buddy::gemmini_ball";
  let summary = "The operations of GemminiBall";
}

class GemminiBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<GemminiBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankMatmulI8TileOp : GemminiBall_Op<"bank_matmul_i8_tile",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_i8_tile on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$k,
                       UnitAttr:$read_destination,
                       OptionalAttr<I64Attr>:$shift);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def MatmulI8TileOp : GemminiBall_Op<"matmul_i8_tile",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_i8_tile on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$k,
                       UnitAttr:$read_destination,
                       OptionalAttr<I64Attr>:$shift);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // GEMMINI_OPS
