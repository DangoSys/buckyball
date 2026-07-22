//===- IntrinsicsRISCVTransposeBall.td - TransposeBall intrinsics --------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_transpose : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
