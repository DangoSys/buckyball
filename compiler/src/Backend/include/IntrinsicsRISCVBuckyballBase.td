//===- IntrinsicsRISCVBuckyballBase.td - Base intrinsics -----------------===//
//
// Common Buckyball intrinsics (mvin, mvout, mset, fence)
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_mvin : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_bb_mvout : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_bb_mset : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;

let TargetPrefix = "riscv" in
def int_riscv_bb_fence : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
