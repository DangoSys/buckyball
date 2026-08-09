func.func private @check_result(memref<576x1024xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %w = memref.alloc() : memref<1024x576xf32>
  %wt = memref.alloc() : memref<576x1024xf32>
  linalg.fill ins(%one : f32) outs(%w : memref<1024x576xf32>)
  linalg.fill ins(%zero : f32) outs(%wt : memref<576x1024xf32>)
  %w_dyn = memref.cast %w : memref<1024x576xf32> to memref<1024x576xf32, strided<[?, ?], offset: ?>>
  tile.tile_transpose %w_dyn %wt
    : memref<1024x576xf32, strided<[?, ?], offset: ?>> memref<576x1024xf32>
  func.call @check_result(%wt) : (memref<576x1024xf32>) -> ()
  memref.dealloc %w : memref<1024x576xf32>
  memref.dealloc %wt : memref<576x1024xf32>
  return %zero_i8 : i8
}
