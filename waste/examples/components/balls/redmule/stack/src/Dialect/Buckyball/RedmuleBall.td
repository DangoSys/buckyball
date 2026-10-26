#ifndef REDMULEBALL_OPS
#define REDMULEBALL_OPS
include "Buckyball.td"

def RedmuleF16Op : Buckyball_Op<"redmule_f16", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Ordered FP16 FMA for A[M,K], B[K,8], C[M,8] bank windows";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows, I64Attr:$k,
    I64Attr:$op1_base, I64Attr:$op2_base, I64Attr:$wr_base,
    BoolAttr:$first);
  let hasVerifier = 1;
  let assemblyFormat = "$a `,` $b `,` $c prop-dict attr-dict `:` type($a)";
}

def BankRedmuleF16Op : Buckyball_Op<"bank_redmule_f16", [DeclareOpInterfaceMethods<PhysicalBankAssignmentInterface>, MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "FP16 bank matrix product with an SSA output bank handle";
  let arguments = (ins I64:$a, I64:$b, I64:$c, I64Attr:$rows, I64Attr:$k,
    I64Attr:$op1_base, I64Attr:$op2_base, I64Attr:$wr_base,
    BoolAttr:$first);
  let results = (outs I64:$cOut);
  let hasVerifier = 1;
  let assemblyFormat = "$a $b $c prop-dict attr-dict `:` type($a) type($b) type($c)";
}
#endif
