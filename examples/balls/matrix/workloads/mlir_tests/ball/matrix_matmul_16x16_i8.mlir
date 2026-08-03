func.func private @check_result(memref<16x16xi32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_i8 = arith.constant 1 : i8
  %zero_i32 = arith.constant 0 : i32
  %a = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
  %b = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
  %c = memref.alloc() {alignment = 64 : i64} : memref<16x16xi32>

  linalg.fill ins(%one_i8 : i8) outs(%a : memref<16x16xi8>)
  linalg.fill ins(%one_i8 : i8) outs(%b : memref<16x16xi8>)
  linalg.fill ins(%zero_i32 : i32) outs(%c : memref<16x16xi32>)
  buckyball.matrix_matmul %a %b %c
      : memref<16x16xi8> memref<16x16xi8> memref<16x16xi32>

  func.call @check_result(%c) : (memref<16x16xi32>) -> ()
  memref.dealloc %a : memref<16x16xi8>
  memref.dealloc %b : memref<16x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %zero_i8 : i8
}
