//===- IntrinsicsRISCVFp2IntBall.td - Fp2IntBall intrinsics --------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_fp2int : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
