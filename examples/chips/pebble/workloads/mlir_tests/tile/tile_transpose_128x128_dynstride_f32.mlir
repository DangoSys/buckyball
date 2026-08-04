func.func private @check_result(memref<128x128xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %w = memref.alloc() : memref<128x128xf32>
  %wt = memref.alloc() : memref<128x128xf32>
  linalg.fill ins(%one : f32) outs(%w : memref<128x128xf32>)
  linalg.fill ins(%zero : f32) outs(%wt : memref<128x128xf32>)
  %w_dyn = memref.cast %w : memref<128x128xf32> to memref<128x128xf32, strided<[?, ?], offset: ?>>
  tile.tile_transpose %w_dyn %wt
    : memref<128x128xf32, strided<[?, ?], offset: ?>> memref<128x128xf32>
  func.call @check_result(%wt) : (memref<128x128xf32>) -> ()
  memref.dealloc %w : memref<128x128xf32>
  memref.dealloc %wt : memref<128x128xf32>
  return %zero_i8 : i8
}
