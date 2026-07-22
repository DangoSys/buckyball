// Buckyball Dialect matmul test: 16x64 @ 64x16
// Matrix: 16x64 (fp32) @ 64x16 (fp32) -> 16x16 (fp32)
// This locks the matrix ISA contract where fp2int, transpose, and
// mul_warp16 process K rows in one instruction.
//
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : i64
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : i64
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i64
// CHECK: buckyball.bank_mvin {{.*}} %[[C64]] %[[C1]]
// CHECK: buckyball.bank_mvin {{.*}} %[[C64]] %[[C1]]
// CHECK: buckyball.bank_fp2int {{.*}} %[[C64]]
// CHECK: buckyball.bank_fp2int {{.*}} %[[C64]]
// CHECK: buckyball.bank_transpose {{.*}} %[[C64]]
// CHECK: buckyball.bank_mul_warp16 {{.*}} %[[C64]]
// CHECK: buckyball.bank_int2fp {{.*}} %[[C16]]
// CHECK: buckyball.bank_mvout {{.*}} %[[C16]] %[[C1]]

func.func private @check_result(memref<16x16xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32

  %a = memref.alloc() {alignment = 16 : i64} : memref<16x64xf32>
  %b = memref.alloc() {alignment = 16 : i64} : memref<64x16xf32>
  %c = memref.alloc() {alignment = 16 : i64} : memref<16x16xf32>

  linalg.fill ins(%one_f32 : f32) outs(%a : memref<16x64xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%b : memref<64x16xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%c : memref<16x16xf32>)

  buckyball.matrix_matmul %a %b %c
    : memref<16x64xf32> memref<64x16xf32> memref<16x16xf32>

  func.call @check_result(%c) : (memref<16x16xf32>) -> ()

  memref.dealloc %a : memref<16x64xf32>
  memref.dealloc %b : memref<64x16xf32>
  memref.dealloc %c : memref<16x16xf32>

  return %zero_i8 : i8
}
