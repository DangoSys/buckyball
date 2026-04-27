//===- IntrinsicsRISCVQuantBall.td - QuantBall intrinsics ----------------===//
//
// QuantBall intrinsics (quant)
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_quant : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
