//===- IntrinsicsRISCVMatrixBall.td - MatrixBall intrinsics --------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_bbfp_mul : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
