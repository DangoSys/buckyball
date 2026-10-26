func.func private @check_result(memref<16x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %c16 = arith.constant 16 : i32
  %input = memref.alloc() alignment = 64 : memref<16x16xi8>
  %output = memref.alloc() alignment = 64 : memref<16x16xi8>

  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %ii = arith.index_cast %i : index to i32
      %jj = arith.index_cast %j : index to i32
      %row_base = arith.muli %ii, %c16 : i32
      %value = arith.addi %row_base, %jj : i32
      %value_i8 = arith.trunci %value : i32 to i8
      memref.store %value_i8, %input[%i, %j] : memref<16x16xi8>
    }
  }

  %bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %loaded = buckyball.bank_mvin %input, %bank <{depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  %stored = buckyball.bank_mvout %output, %loaded <{depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.bank_release %stored

  func.call @check_result(%output) : (memref<16x16xi8>) -> ()
  memref.dealloc %input : memref<16x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero_i8 : i8
}
