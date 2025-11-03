// RUN: buddy-opt %s \
// RUN:   -convert-linalg-to-tile \
// RUN:   -convert-tile-to-buckyball \
// RUN:   -lower-buckyball="dim=16 sp_addr_len=14 spad_rows=1024 acc_rows=1024 warp=16 lane=16" \
// RUN: | FileCheck %s

// Complete conversion flow test: linalg.matmul → tile.tile_matmul → buckyball.bb_matmul → intrinsics
func.func @end_to_end_test(%arg0: memref<32x32xi8>, %arg1: memref<32x32xi8>, %arg2: memref<32x32xi32>) {
  // CHECK: buckyball.intr.bb_mvin
  // CHECK: buckyball.intr.bb_mul_warp16
  // CHECK: buckyball.intr.bb_mvout
  linalg.matmul
    ins(%arg0, %arg1 : memref<32x32xi8>, memref<32x32xi8>)
    outs(%arg2 : memref<32x32xi32>)
  return
}
