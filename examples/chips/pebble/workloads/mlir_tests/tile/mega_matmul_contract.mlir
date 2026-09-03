// CHECK-LABEL: func.func @lenet_matmul_chain(
// CHECK: buckyball.mega_kernel
// CHECK-NEXT: buckyball.mega_matmul
// CHECK-SAME: {activation = 1 : i64}
// CHECK-NEXT: buckyball.mega_matmul
// CHECK-SAME: {activation = 1 : i64}
// CHECK-NEXT: buckyball.mega_matmul
// CHECK-SAME: {activation = 0 : i64}
// CHECK-NEXT: }
// CHECK-NOT: tile.mega_kernel
func.func @lenet_matmul_chain(
    %input: memref<1x256xi8>,
    %weight0: memref<256x120xi8>,
    %bias0: memref<120xi32>,
    %scale0: memref<120xf32>,
    %lut0: memref<1xi8>,
    %hidden0: memref<1x120xi8>,
    %weight1: memref<120x84xi8>,
    %bias1: memref<84xi32>,
    %scale1: memref<84xf32>,
    %lut1: memref<1xi8>,
    %hidden1: memref<1x84xi8>,
    %weight2: memref<84x10xi8>,
    %bias2: memref<10xi32>,
    %scale2: memref<10xf32>,
    %lut2: memref<1xi8>,
    %output: memref<1x10xf32>) {
  tile.mega_kernel %input %output : memref<1x256xi8> memref<1x10xf32> {
    tile.mega_matmul %input %weight0 %bias0 %scale0 %lut0 %hidden0 {activation = 1 : i64} :
      memref<1x256xi8> memref<256x120xi8> memref<120xi32>
      memref<120xf32> memref<1xi8> memref<1x120xi8>
    tile.mega_matmul %hidden0 %weight1 %bias1 %scale1 %lut1 %hidden1 {activation = 1 : i64} :
      memref<1x120xi8> memref<120x84xi8> memref<84xi32>
      memref<84xf32> memref<1xi8> memref<1x84xi8>
    tile.mega_matmul %hidden1 %weight2 %bias2 %scale2 %lut2 %output {activation = 0 : i64} :
      memref<1x84xi8> memref<84x10xi8> memref<10xi32>
      memref<10xf32> memref<1xi8> memref<1x10xf32>
  }
  return
}
