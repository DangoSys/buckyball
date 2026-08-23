func.func private @check_result(memref<80x80xf32>, memref<80x80xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c80 = arith.constant 80 : index
  %w = memref.alloc() : memref<80x80xf32>
  %wt = memref.alloc() : memref<80x80xf32>
  linalg.fill ins(%zero : f32) outs(%wt : memref<80x80xf32>)
  scf.for %i = %c0 to %c80 step %c1 {
    scf.for %j = %c0 to %c80 step %c1 {
      %ii = arith.index_cast %i : index to i32
      %jj = arith.index_cast %j : index to i32
      %s = arith.addi %ii, %jj : i32
      %v = arith.sitofp %s : i32 to f32
      memref.store %v, %w[%i, %j] : memref<80x80xf32>
    }
  }
  tile.tile_transpose %w %wt : memref<80x80xf32> memref<80x80xf32>
  func.call @check_result(%w, %wt) : (memref<80x80xf32>, memref<80x80xf32>) -> ()
  memref.dealloc %w : memref<80x80xf32>
  memref.dealloc %wt : memref<80x80xf32>
  return %zero_i8 : i8
}
