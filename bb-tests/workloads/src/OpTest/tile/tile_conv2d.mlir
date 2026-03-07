// Test for tile-level conv2d: linalg.conv_2d_nhwc_hwcf -> tile.tile_conv2d -> buckyball
// Pipeline: -convert-linalg-to-tile -convert-tile-to-buckyball -lower-buckyball
// Input: [1,6,6,1], Filter: [3,3,1,1], Output: [1,4,4,1]

// Input image: 6x6 with incrementing values
memref.global "private" @input : memref<1x6x6x1xi8> = dense<[[
  [[1],[2],[3],[4],[5],[6]],
  [[7],[8],[9],[10],[11],[12]],
  [[13],[14],[15],[16],[17],[18]],
  [[19],[20],[21],[22],[23],[24]],
  [[25],[26],[27],[28],[29],[30]],
  [[31],[32],[33],[34],[35],[36]]
]]>

// Filter: 3x3 all ones (sum pooling equivalent)
memref.global "private" @filter : memref<3x3x1x1xi8> = dense<[[
  [[1]],[[1]],[[1]]
],[
  [[1]],[[1]],[[1]]
],[
  [[1]],[[1]],[[1]]
]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8

  %input = memref.get_global @input : memref<1x6x6x1xi8>
  %filter = memref.get_global @filter : memref<3x3x1x1xi8>
  %output = memref.alloc() : memref<1x4x4x1xi32>

  // linalg.conv_2d_nhwc_hwcf -> tile.tile_conv2d -> im2col + matmul
  linalg.conv_2d_nhwc_hwcf
    ins(%input, %filter : memref<1x6x6x1xi8>, memref<3x3x1x1xi8>)
    outs(%output : memref<1x4x4x1xi32>)

  buckyball.bb_print_memref %output : memref<1x4x4x1xi32>

  memref.dealloc %output : memref<1x4x4x1xi32>
  return %0 : i8
}
