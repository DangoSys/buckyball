// Buckyball Dialect matmul test: 1x128 @ 128x96
// Matrix: 1x128 (fp32) @ 128x96 (fp32) -> 1x96 (fp32)
// This mirrors the small-M, wide-N shape seen in the LeNet FC tail.

func.func private @check_result(memref<1x96xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32

  %a = memref.alloc() {alignment = 16 : i64} : memref<1x128xf32>
  %b = memref.alloc() {alignment = 16 : i64} : memref<128x96xf32>
  %c = memref.alloc() {alignment = 16 : i64} : memref<1x96xf32>

  linalg.fill ins(%one_f32 : f32) outs(%a : memref<1x128xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%b : memref<128x96xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%c : memref<1x96xf32>)

  buckyball.matmul %a %b %c
    : memref<1x128xf32> memref<128x96xf32> memref<1x96xf32>

  func.call @check_result(%c) : (memref<1x96xf32>) -> ()

  memref.dealloc %a : memref<1x128xf32>
  memref.dealloc %b : memref<128x96xf32>
  memref.dealloc %c : memref<1x96xf32>

  return %zero_i8 : i8
}
