//===- IntrinsicsRISCVDequantBall.td - DequantBall intrinsics ------------===//
//
// DequantBall intrinsics (dequant)
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in
def int_riscv_bb_int2fp : Intrinsic<[], [llvm_i64_ty, llvm_i64_ty], []>;
