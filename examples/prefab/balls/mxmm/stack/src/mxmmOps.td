//===- mxmmOps.td - MxmmBall operations --------------------*- tablegen -*-===//
//
// matmul_<format> (a, b) -> c: one step of a chained FP32 accumulation
// C[rows x cols] += A[rows x k] @ B[k x cols]. `rows` is one or a multiple of
// sixteen, `cols` a multiple of sixteen. The chain lives in the Ball, one per
// hart: `first` starts it from +0, the steps between continue it, `last`
// writes it to C as `rows` x `cols` FP32 in row-major order from row `base`
// (0..63) of the bank; nothing else of C is written, and no step reads C. A
// step without `first` continues the live chain with the same format, rows,
// cols, C and base.
//   mxfp8: k is a multiple of 32. A holds rows x k E4M3 codes row-major, then
//     the rows x k/32 E8M0 scales row-major. B holds the cols x k codes, column
//     c the k codes from byte c * k, then the cols x k/32 scales, column c from
//     byte cols * k + c * k/32.
//   mxfp8_window: mxfp8 on columns start_k .. start_k + k of an A of
//     full_k columns; full_k and start_k are multiples of 32.
//   fma32: k is a multiple of four. A holds rows x k FP32 row-major, B column c
//     the k FP32 from byte 4 * c * k; every product is fused into the sum.
//   f32: fma32 with the product rounded before the sum.
// A, B and C are distinct banks of one 128-bit wide bank each.
//
//===----------------------------------------------------------------------===//

#ifndef MXMM_OPS
#define MXMM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Interfaces/BankInterfaces.td"

def MxmmBall_Dialect : Dialect {
  let name = "mxmm_ball";
  let cppNamespace = "::buddy::mxmm_ball";
  let summary = "The operations of MxmmBall";
}

class MxmmBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<MxmmBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankMatmulMxfp8Op : MxmmBall_Op<"bank_matmul_mxfp8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_mxfp8 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows, I64Attr:$cols,
                       I64Attr:$k, I64Attr:$base, UnitAttr:$first,
                       UnitAttr:$last);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def MatmulMxfp8Op : MxmmBall_Op<"matmul_mxfp8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_mxfp8 on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$rows,
                       I64Attr:$cols, I64Attr:$k, I64Attr:$base,
                       UnitAttr:$first, UnitAttr:$last);
  let assemblyFormat = "prop-dict attr-dict";
}

def BankMatmulMxfp8WindowOp : MxmmBall_Op<"bank_matmul_mxfp8_window",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_mxfp8_window on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows, I64Attr:$cols,
                       I64Attr:$k, I64Attr:$full_k, I64Attr:$start_k,
                       I64Attr:$base, UnitAttr:$first, UnitAttr:$last);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def MatmulMxfp8WindowOp : MxmmBall_Op<"matmul_mxfp8_window",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_mxfp8_window on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$rows,
                       I64Attr:$cols, I64Attr:$k, I64Attr:$full_k,
                       I64Attr:$start_k, I64Attr:$base, UnitAttr:$first,
                       UnitAttr:$last);
  let assemblyFormat = "prop-dict attr-dict";
}

def BankMatmulFma32Op : MxmmBall_Op<"bank_matmul_fma32",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_fma32 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows, I64Attr:$cols,
                       I64Attr:$k, I64Attr:$base, UnitAttr:$first,
                       UnitAttr:$last);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def MatmulFma32Op : MxmmBall_Op<"matmul_fma32",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_fma32 on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$rows,
                       I64Attr:$cols, I64Attr:$k, I64Attr:$base,
                       UnitAttr:$first, UnitAttr:$last);
  let assemblyFormat = "prop-dict attr-dict";
}

def BankMatmulF32Op : MxmmBall_Op<"bank_matmul_f32",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_f32 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows, I64Attr:$cols,
                       I64Attr:$k, I64Attr:$base, UnitAttr:$first,
                       UnitAttr:$last);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def MatmulF32Op : MxmmBall_Op<"matmul_f32",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_f32 on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$rows,
                       I64Attr:$cols, I64Attr:$k, I64Attr:$base,
                       UnitAttr:$first, UnitAttr:$last);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MXMM_OPS
