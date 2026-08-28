func.func private @check_result(memref<16x16xi8>) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %input = memref.alloc() alignment = 64 : memref<16x16xi8>
  %output = memref.alloc() alignment = 64 : memref<16x16xi8>
  scf.for %i = %c0 to %c16 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %ii = arith.index_cast %i : index to i8
      %jj = arith.index_cast %j : index to i8
      %v = arith.addi %ii, %jj : i8
      memref.store %v, %input[%i, %j] : memref<16x16xi8>
    }
  }
  tile.tile_transpose %input %output : memref<16x16xi8> memref<16x16xi8>
  func.call @check_result(%output) : (memref<16x16xi8>) -> ()
  memref.dealloc %input : memref<16x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero : i8
}
