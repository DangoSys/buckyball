//===- mxint8mmOps.td - Mxint8mmBall operations ------------*- tablegen -*-===//
//
// matmul_mxint8 (a, b) -> c: C[m, n] (FP32, row-major, n/4 bank rows per
// row) = A[m, k] . B[n, k]^T, one MXINT8MM instruction. A and B hold packed
// MXINT8 rows along k (a multiple of 32): k/16 + ceil(k/512) bank rows each,
// the k INT8 codes then the k/32 exponent bytes. Every operand starts at
// bank row 0: the bank-row fields of the instruction are 0. Per element and block b in order:
// t = fp32(sum of the 32 code products * 2^(ea + eb)), acc = fp32(acc + t),
// acc starting at +0.0, or at the old C with `accumulate`. Terms and sums
// must be finite. n is a multiple of 4; a, b and c are distinct banks of one
// 128-bit wide group each.
//
//===----------------------------------------------------------------------===//

#ifndef MXINT8MM_OPS
#define MXINT8MM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Acc/BankInterfaces.td"

def Mxint8mmBall_Dialect : Dialect {
  let name = "mxint8mm_ball";
  let cppNamespace = "::buddy::mxint8mm_ball";
  let summary = "The operations of Mxint8mmBall";
}

class Mxint8mmBall_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mxint8mmBall_Dialect, mnemonic,
         !listconcat(traits, [MemoryEffects<[MemRead, MemWrite]>])>;

def BankMatmulMxint8Op : Mxint8mmBall_Op<"bank_matmul_mxint8",
    [DeclareOpInterfaceMethods<BankSSALowering>]> {
  let summary = "matmul_mxint8 on virtual banks";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$m, I64Attr:$n,
                   I64Attr:$k, BoolAttr:$accumulate);
  let results = (outs I64:$cOut);
  let assemblyFormat = "$a `,` $b `->` $c prop-dict attr-dict";
}

def MatmulMxint8Op : Mxint8mmBall_Op<"matmul_mxint8",
    [DeclareOpInterfaceMethods<PhysicalBankAssignment>]> {
  let summary = "matmul_mxint8 on physical banks";
  let arguments = (ins I64Attr:$a, I64Attr:$b, I64Attr:$c, I64Attr:$m,
                   I64Attr:$n, I64Attr:$k, BoolAttr:$accumulate);
  let assemblyFormat = "prop-dict attr-dict";
}

#endif // MXINT8MM_OPS
