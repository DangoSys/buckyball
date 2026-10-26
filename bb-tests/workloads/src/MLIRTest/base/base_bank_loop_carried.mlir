func.func private @check_result(memref<64x16xi8>) -> ()

// One bank carried through a loop: each iteration moves 16 rows through it.
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %input = memref.alloc() alignment = 64 : memref<64x16xi8>
  %output = memref.alloc() alignment = 64 : memref<64x16xi8>

  scf.for %i = %c0 to %c64 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %ii = arith.index_cast %i : index to i32
      %jj = arith.index_cast %j : index to i32
      %sum = arith.addi %ii, %jj : i32
      %value = arith.trunci %sum : i32 to i8
      memref.store %value, %input[%i, %j] : memref<64x16xi8>
    }
  }

  %bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %last = scf.for %row = %c0 to %c64 step %c16 iter_args(%carried = %bank) -> (i64) {
    %in = memref.subview %input[%row, 0] [16, 16] [1, 1]
        : memref<64x16xi8> to memref<16x16xi8, strided<[16, 1], offset: ?>>
    %out = memref.subview %output[%row, 0] [16, 16] [1, 1]
        : memref<64x16xi8> to memref<16x16xi8, strided<[16, 1], offset: ?>>
    %loaded = buckyball.bank_mvin %in, %carried <{depth = 16 : i64, stride = 1 : i64}>
        : memref<16x16xi8, strided<[16, 1], offset: ?>>
    %stored = buckyball.bank_mvout %out, %loaded <{depth = 16 : i64, stride = 1 : i64}>
        : memref<16x16xi8, strided<[16, 1], offset: ?>>
    scf.yield %stored : i64
  }
  buckyball.bank_release %last

  func.call @check_result(%output) : (memref<64x16xi8>) -> ()
  memref.dealloc %input : memref<64x16xi8>
  memref.dealloc %output : memref<64x16xi8>
  return %zero_i8 : i8
}
