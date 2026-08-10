func.func private @check_result(memref<5x768xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %a = memref.alloc() : memref<5x3072xf32>
  %b = memref.alloc() : memref<3072x768xf32>
  %c = memref.alloc() : memref<5x768xf32>
  linalg.fill ins(%one : f32) outs(%a : memref<5x3072xf32>)
  linalg.fill ins(%one : f32) outs(%b : memref<3072x768xf32>)
  linalg.fill ins(%zero : f32) outs(%c : memref<5x768xf32>)
  tile.tile_matmul %a %b %c : memref<5x3072xf32> memref<3072x768xf32> memref<5x768xf32>
  func.call @check_result(%c) : (memref<5x768xf32>) -> ()
  memref.dealloc %a : memref<5x3072xf32>
  memref.dealloc %b : memref<3072x768xf32>
  memref.dealloc %c : memref<5x768xf32>
  return %zero_i8 : i8
}
