func.func private @check_result(memref<1x1024xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %a = memref.alloc() : memref<1x576xf32>
  %b = memref.alloc() : memref<576x1024xf32>
  %c = memref.alloc() : memref<1x1024xf32>
  linalg.fill ins(%one : f32) outs(%a : memref<1x576xf32>)
  linalg.fill ins(%one : f32) outs(%b : memref<576x1024xf32>)
  linalg.fill ins(%zero : f32) outs(%c : memref<1x1024xf32>)
  tile.tile_matmul %a %b %c : memref<1x576xf32> memref<576x1024xf32> memref<1x1024xf32>
  func.call @check_result(%c) : (memref<1x1024xf32>) -> ()
  memref.dealloc %a : memref<1x576xf32>
  memref.dealloc %b : memref<576x1024xf32>
  memref.dealloc %c : memref<1x1024xf32>
  return %zero_i8 : i8
}
