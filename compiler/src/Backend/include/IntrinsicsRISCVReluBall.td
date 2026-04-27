//===- IntrinsicsRISCVReluBall.td - ReluBall intrinsics ------------------===//
//
// ReluBall intrinsics (relu)
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_relu : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
