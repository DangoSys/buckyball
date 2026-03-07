// Test for conv2d end-to-end: tile_conv2d -> im2col + matmul
// RUN: buddy-opt %s \
// RUN:     -convert-tile-to-buckyball \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec:
// Purpose: verify conv2d tiling and lowering through Tile -> Buckyball -> LLVM
// Input: [1, 4, 4, 1] (N=1, H=4, W=4, C=1)
// Filter: [3, 3, 1, 1] (KH=3, KW=3, C=1, OC=1)
// Output: [1, 2, 2, 1] (N=1, OH=2, OW=2, OC=1)

// Input: 1x4x4x1
memref.global "private" @conv_input : memref<1x4x4x1xi8> = dense<[[
  [[1], [2], [3], [4]],
  [[5], [6], [7], [8]],
  [[9], [10], [11], [12]],
  [[13], [14], [15], [16]]
]]>

// Filter: 3x3x1x1
memref.global "private" @conv_filter : memref<3x3x1x1xi8> = dense<[[
  [[1]], [[0]], [[0]],
  [[0]], [[1]], [[0]],
  [[0]], [[0]], [[1]]
]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8

  %input = memref.get_global @conv_input : memref<1x4x4x1xi8>
  %filter = memref.get_global @conv_filter : memref<3x3x1x1xi8>
  %output = memref.alloc() : memref<1x2x2x1xi8>

  // Print input
  // buckyball.print %input : memref<1x4x4x1xi8>

  // CHECK: tile_conv2d
  tile.tile_conv2d %input %filter %output : memref<1x4x4x1xi8> memref<3x3x1x1xi8> memref<1x2x2x1xi8>

  memref.dealloc %output : memref<1x2x2x1xi8>
  return %0 : i8
}
