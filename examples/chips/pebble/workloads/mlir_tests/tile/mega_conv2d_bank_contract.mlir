// CHECK-LABEL: func.func @main(
// CHECK: buckyball.transpose
// CHECK: buckyball.im2col
// CHECK: buckyball.smatmul
// CHECK: buckyball.quant_i32_to_i8
// CHECK: buckyball.int32_to_fp32
// CHECK-NOT: buckyball.mega_kernel
// CHECK-NOT: buckyball.bank_

func.func @main(
    %input: memref<1x8x8x16xi8>,
    %expandWeight: memref<1x1x16x32xi8>,
    %expandBias: memref<32xi32>,
    %expandScale: memref<32xf32>,
    %expanded: memref<1x8x8x32xi8>,
    %depthwiseWeight: memref<3x3x32x1xi8>,
    %depthwiseBias: memref<32xi32>,
    %depthwiseScale: memref<32xf32>,
    %depthwiseOut: memref<1x8x8x32xi8>,
    %projectWeight: memref<1x1x32x16xi8>,
    %projectBias: memref<16xi32>,
    %projectScale: memref<16xf32>,
    %output: memref<1x16x8x8xf32>) {
  tile.mega_kernel %input %output
      : memref<1x8x8x16xi8> memref<1x16x8x8xf32> {
    tile.mega_conv2d %input %expandWeight %expandBias %expandScale %expanded {
      stride = 1 : i64, padLow = 0 : i64, padHigh = 0 : i64, relu = true
    } : memref<1x8x8x16xi8> memref<1x1x16x32xi8> memref<32xi32>
        memref<32xf32> memref<1x8x8x32xi8>
    tile.mega_conv2d_depthwise %expanded %depthwiseWeight %depthwiseBias
        %depthwiseScale %depthwiseOut {
      stride = 1 : i64, padLow = 1 : i64, padHigh = 1 : i64, relu = true
    } : memref<1x8x8x32xi8> memref<3x3x32x1xi8> memref<32xi32>
        memref<32xf32> memref<1x8x8x32xi8>
    tile.mega_conv2d %depthwiseOut %projectWeight %projectBias %projectScale
        %output {
      stride = 1 : i64, padLow = 0 : i64, padHigh = 0 : i64, relu = false
    } : memref<1x8x8x32xi8> memref<1x1x32x16xi8> memref<16xi32>
        memref<16xf32> memref<1x16x8x8xf32>
  }
  return
}
