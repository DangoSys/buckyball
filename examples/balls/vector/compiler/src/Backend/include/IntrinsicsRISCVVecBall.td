//===- IntrinsicsRISCVVecBall.td - VecBall intrinsics --------------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_mul_warp16 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
