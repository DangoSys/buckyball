// CHECK-LABEL: func.func @mega_conv2d(
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 17 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 48 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 65 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 48 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 65 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 67 : i32
// CHECK-NOT: tile.mega_conv2d
// CHECK-NOT: buckyball.mega_conv2d
// CHECK-NOT: buckyball.bank_

func.func @mega_conv2d(
    %input: memref<1x8x8x2xi8>,
    %weight: memref<3x3x2x16xi8>,
    %bias: memref<16xi32>,
    %scale: memref<16xf32>,
    %output: memref<1x8x8x16xi8>) {
  tile.mega_conv2d %input %weight %bias %scale %output {
    stride = 1 : i64, padLow = 1 : i64, padHigh = 1 : i64,
    relu = true, depthwise = false
  } : memref<1x8x8x2xi8> memref<3x3x2x16xi8> memref<16xi32>
      memref<16xf32> memref<1x8x8x16xi8>
  return
}
