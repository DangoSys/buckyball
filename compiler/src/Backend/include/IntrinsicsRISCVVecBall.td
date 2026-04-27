//===- IntrinsicsRISCVVecBall.td - VecBall intrinsics --------------------===//
//
// VecBall intrinsics (mul_warp16)
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_mul_warp16 : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
