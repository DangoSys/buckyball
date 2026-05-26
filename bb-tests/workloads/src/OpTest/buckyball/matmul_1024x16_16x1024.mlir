// Buckyball Dialect matmul test: 1024x16 @ 16x1024

func.func private @check_result(memref<1024x1024xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32

  %a = memref.alloc() : memref<1024x16xf32>
  %b = memref.alloc() : memref<16x1024xf32>
  %c = memref.alloc() : memref<1024x1024xf32>

  linalg.fill ins(%one_f32 : f32) outs(%a : memref<1024x16xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%b : memref<16x1024xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%c : memref<1024x1024xf32>)

  buckyball.matmul %a %b %c
    : memref<1024x16xf32> memref<16x1024xf32> memref<1024x1024xf32>

  func.call @check_result(%c) : (memref<1024x1024xf32>) -> ()

  memref.dealloc %a : memref<1024x16xf32>
  memref.dealloc %b : memref<16x1024xf32>
  memref.dealloc %c : memref<1024x1024xf32>

  return %zero_i8 : i8
}
