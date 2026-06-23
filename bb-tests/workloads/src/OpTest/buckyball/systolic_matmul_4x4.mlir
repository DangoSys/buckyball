// Prototype SystolicArrayBall contract: 4x4 i8 @ 4x4 i8 -> 4x4 i32.
// The lowering must materialize explicit packed bank-line buffers and use
// bank_systolic rather than bank_mul_warp16.
//
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i64
// CHECK: %[[PA:.*]] = memref.alloc() {{.*}} : memref<4x16xi8>
// CHECK: %[[PB:.*]] = memref.alloc() {{.*}} : memref<4x16xi8>
// CHECK: %[[PC:.*]] = memref.alloc() {{.*}} : memref<4x16xi32>
// CHECK: buckyball.bank_mvin {{.*}} %[[C4]]
// CHECK: buckyball.bank_mvin {{.*}} %[[C4]]
// CHECK-DAG: %[[CFG_OS:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[CFG_WS:.*]] = arith.constant 1 : i64
// CHECK: buckyball.bank_systolic {{.*}} %[[C4]] %[[CFG_OS]]
// CHECK: buckyball.bank_mvout {{.*}} %[[C4]]
// CHECK: buckyball.bank_systolic {{.*}} %[[C4]] %[[CFG_WS]]

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_i8 = arith.constant 1 : i8
  %zero_i32 = arith.constant 0 : i32
  %a = memref.alloc() {alignment = 16 : i64} : memref<4x4xi8>
  %b = memref.alloc() {alignment = 16 : i64} : memref<4x4xi8>
  %c = memref.alloc() {alignment = 16 : i64} : memref<4x4xi32>
  %c_ws = memref.alloc() {alignment = 16 : i64} : memref<4x4xi32>

  linalg.fill ins(%one_i8 : i8) outs(%a : memref<4x4xi8>)
  linalg.fill ins(%one_i8 : i8) outs(%b : memref<4x4xi8>)
  linalg.fill ins(%zero_i32 : i32) outs(%c : memref<4x4xi32>)
  linalg.fill ins(%zero_i32 : i32) outs(%c_ws : memref<4x4xi32>)

  buckyball.systolic_matmul %a %b %c
    : memref<4x4xi8> memref<4x4xi8> memref<4x4xi32>
  buckyball.systolic_matmul %a %b %c_ws {ws = true}
    : memref<4x4xi8> memref<4x4xi8> memref<4x4xi32>

  memref.dealloc %a : memref<4x4xi8>
  memref.dealloc %b : memref<4x4xi8>
  memref.dealloc %c : memref<4x4xi32>
  memref.dealloc %c_ws : memref<4x4xi32>

  return %zero_i8 : i8
}
