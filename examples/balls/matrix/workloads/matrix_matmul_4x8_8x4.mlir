// Prototype MatrixBall contract: 4x8 i8 @ 8x4 i8 -> 4x4 i32.
// The lowering must split the K dimension into two 4-wide matrix commands
// and use config[2:1] for cross-instruction accumulation.
//
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[CFG_FIRST_OS:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[CFG_LAST_OS:.*]] = arith.constant 6 : i64
// CHECK-DAG: %[[CFG_FIRST_WS:.*]] = arith.constant 3 : i64
// CHECK-DAG: %[[CFG_LAST_WS:.*]] = arith.constant 7 : i64
// CHECK: buckyball.bank_matrix {{.*}} %[[C4]] %[[CFG_FIRST_OS]]
// CHECK: buckyball.bank_matrix {{.*}} %[[C4]] %[[CFG_LAST_OS]]
// CHECK: buckyball.bank_mvout {{.*}} %[[C4]]
// CHECK: buckyball.bank_matrix {{.*}} %[[C4]] %[[CFG_FIRST_WS]]
// CHECK: buckyball.bank_matrix {{.*}} %[[C4]] %[[CFG_LAST_WS]]
// CHECK: buckyball.bank_mvout {{.*}} %[[C4]]

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_i8 = arith.constant 1 : i8
  %zero_i32 = arith.constant 0 : i32
  %a = memref.alloc() {alignment = 16 : i64} : memref<4x8xi8>
  %b = memref.alloc() {alignment = 16 : i64} : memref<8x4xi8>
  %c = memref.alloc() {alignment = 16 : i64} : memref<4x4xi32>
  %c_ws = memref.alloc() {alignment = 16 : i64} : memref<4x4xi32>

  linalg.fill ins(%one_i8 : i8) outs(%a : memref<4x8xi8>)
  linalg.fill ins(%one_i8 : i8) outs(%b : memref<8x4xi8>)
  linalg.fill ins(%zero_i32 : i32) outs(%c : memref<4x4xi32>)
  linalg.fill ins(%zero_i32 : i32) outs(%c_ws : memref<4x4xi32>)

  buckyball.matrix_matmul %a %b %c
    : memref<4x8xi8> memref<8x4xi8> memref<4x4xi32>
  buckyball.matrix_matmul %a %b %c_ws {ws = true}
    : memref<4x8xi8> memref<8x4xi8> memref<4x4xi32>

  memref.dealloc %a : memref<4x8xi8>
  memref.dealloc %b : memref<8x4xi8>
  memref.dealloc %c : memref<4x4xi32>
  memref.dealloc %c_ws : memref<4x4xi32>

  return %zero_i8 : i8
}
