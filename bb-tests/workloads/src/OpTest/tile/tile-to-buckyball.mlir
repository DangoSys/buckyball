// RUN: buddy-opt %s -convert-tile-to-buckyball | FileCheck %s

func.func @tile_to_buckyball(%arg0: memref<32x32xi8>, %arg1: memref<32x32xi8>, %arg2: memref<32x32xi32>) {
  // CHECK: buckyball.bb_matmul
  tile.tile_matmul %arg0 %arg1 %arg2 : memref<32x32xi8> memref<32x32xi8> memref<32x32xi32>
  return
}
