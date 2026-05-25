//===- IntrinsicsRISCVQuantBall.td - QuantBall intrinsics ----------------===//
//
// QuantBall intrinsics (quant)
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_fp2int : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
