//===- IntrinsicsRISCVIm2colBall.td - Im2colBall intrinsics --------------===//
//
// Im2colBall intrinsics (im2col)
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_im2col : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
