// CHECK-LABEL: func.func @conv_depthwise(
// CHECK: buckyball.mega_kernel
// CHECK-NEXT: buckyball.mega_conv2d
// CHECK-SAME: {padHigh = 1 : i64, padLow = 1 : i64, relu = true, stride = 1 : i64}
// CHECK-NEXT: buckyball.mega_conv2d_depthwise
// CHECK-SAME: {padHigh = 1 : i64, padLow = 1 : i64, relu = false, stride = 1 : i64}
// CHECK-NEXT: }
// CHECK-NOT: tile.mega_kernel
func.func @conv_depthwise(
    %input: memref<1x8x8x2xi8>,
    %weight0: memref<3x3x2x16xi8>,
    %bias0: memref<16xi32>,
    %scale0: memref<16xf32>,
    %intermediate: memref<1x8x8x16xi8>,
    %weight1: memref<3x3x16x1xi8>,
    %bias1: memref<16xi32>,
    %scale1: memref<16xf32>,
    %output: memref<1x16x8x8xf32>) {
  tile.mega_kernel %input %output : memref<1x8x8x2xi8> memref<1x16x8x8xf32> {
    tile.mega_conv2d %input %weight0 %bias0 %scale0 %intermediate {
      stride = 1 : i64, padLow = 1 : i64, padHigh = 1 : i64, relu = true
    } : memref<1x8x8x2xi8> memref<3x3x2x16xi8> memref<16xi32>
        memref<16xf32> memref<1x8x8x16xi8>
    tile.mega_conv2d_depthwise %intermediate %weight1 %bias1 %scale1 %output {
      stride = 1 : i64, padLow = 1 : i64, padHigh = 1 : i64, relu = false
    } : memref<1x8x8x16xi8> memref<3x3x16x1xi8> memref<16xi32>
        memref<16xf32> memref<1x16x8x8xf32>
  }
  return
}
