//===- mxint8matmulOps.td - Mxint8matmulBall operations ----*- tablegen -*-===//
//
// matmul_mxint8 (a, b) -> c: C[m, n] (FP32, n/4 bank rows per row) = A[m, k]
// . B[n, k]^T, one MXINT8MM instruction. A and B hold packed MXINT8 rows
// along k (a multiple of 32), row r at bank row r * (k / 16 + ceil(k / 512)):
// the k INT8 codes, then the k / 32 exponent bytes. Per element and block b
// in order: t = fp32(sum of the 32 code products * 2^(ea + eb)) (rounded to
// nearest even), acc = fp32(acc + t), acc starting at +0.0, or at the old C
// with `accumulate`. Terms, sums and the old C must be finite. n is a
// multiple of 4; a, b and c are distinct banks of one 128-bit wide group
// each, every operand starting at bank row 0.
//
//===----------------------------------------------------------------------===//

#ifndef MXINT8MATMUL_OPS
#define MXINT8MATMUL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def Mxint8matmulBall_Dialect : Dialect {
  let name = "mxint8matmul_ball";
  let cppNamespace = "::buddy::mxint8matmul_ball";
  let summary = "The operations of Mxint8matmulBall";
}

class Mxint8matmulBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mxint8matmulBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankMatmulMxint8Op : Mxint8matmulBall_Op<"bank_matmul_mxint8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_mxint8 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$m, I64Attr:$n,
                   I64Attr:$k, BoolAttr:$accumulate);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def MatmulMxint8Op : Mxint8matmulBall_Op<"matmul_mxint8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_mxint8 on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$m,
                   I64Attr:$n, I64Attr:$k, BoolAttr:$accumulate);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MXINT8MATMUL_OPS
