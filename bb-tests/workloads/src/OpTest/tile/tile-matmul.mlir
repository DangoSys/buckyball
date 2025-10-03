// RUN: buddy-opt %s -convert-linalg-to-tile | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul_tile(%arg0: memref<32x32xi8>, %arg1: memref<32x32xi8>, %arg2: memref<32x32xi32>) {
  // CHECK: tile.tile_matmul
  linalg.matmul
    ins(%arg0, %arg1 : memref<32x32xi8>, memref<32x32xi8>)
    outs(%arg2 : memref<32x32xi32>)
  return
}
