func.func private @check_result(memref<80x80xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %w = memref.alloc() : memref<80x80xf32>
  %wt = memref.alloc() : memref<80x80xf32>
  linalg.fill ins(%one : f32) outs(%w : memref<80x80xf32>)
  linalg.fill ins(%zero : f32) outs(%wt : memref<80x80xf32>)
  tile.tile_transpose %w %wt : memref<80x80xf32> memref<80x80xf32>
  func.call @check_result(%wt) : (memref<80x80xf32>) -> ()
  memref.dealloc %w : memref<80x80xf32>
  memref.dealloc %wt : memref<80x80xf32>
  return %zero_i8 : i8
}
