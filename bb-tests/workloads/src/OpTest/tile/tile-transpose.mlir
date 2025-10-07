// RUN: buddy-opt %s -convert-linalg-to-tile | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

func.func @transpose_tile(%arg0: memref<32x32xi8>, %arg1: memref<32x32xi8>) {
  // CHECK: tile.tile_transpose
  linalg.transpose
    ins(%arg0 : memref<32x32xi8>)
    outs(%arg1 : memref<32x32xi8>)
    permutation = [1, 0]
  return
}
