// CHECK-LABEL: func.func @mega_matmul(
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 17 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 65 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 65 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 67 : i32
// CHECK-NOT: tile.mega_matmul
// CHECK-NOT: buckyball.mega_matmul
// CHECK-NOT: buckyball.bank_

func.func @mega_matmul(
    %input: memref<16x512xi8>,
    %weight: memref<512x16xi8>,
    %bias: memref<16xi32>,
    %scale: memref<16xf32>,
    %output: memref<16x16xi8>) {
  tile.mega_matmul %input %weight %bias %scale %output {relu = true} :
    memref<16x512xi8> memref<512x16xi8> memref<16xi32>
    memref<16xf32> memref<16x16xi8>
  return
}
