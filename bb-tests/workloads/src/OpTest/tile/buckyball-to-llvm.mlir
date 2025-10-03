// RUN: buddy-opt %s -lower-buckyball="dim=16 sp_addr_len=14 spad_rows=1024 acc_rows=1024 warp=16 lane=16" | FileCheck %s

func.func @buckyball_matmul_to_llvm(%arg0: memref<32x16xi8>, %arg1: memref<16x32xi8>, %arg2: memref<32x32xi32>) {
  // CHECK: buckyball.intr.bb_mvin
  // CHECK: buckyball.intr.bb_mvin
  // CHECK: buckyball.intr.bb_mul_warp16
  // CHECK: buckyball.intr.bb_mvout
  buckyball.bb_matmul %arg0 %arg1 %arg2 : memref<32x16xi8> memref<16x32xi8> memref<32x32xi32>
  return
}
