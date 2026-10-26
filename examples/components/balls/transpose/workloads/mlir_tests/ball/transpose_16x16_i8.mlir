// Physical-bank transpose_i8 of a 16x16 i8 matrix.

func.func private @check_result(memref<16x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %c16_i32 = arith.constant 16 : i32

  %input = memref.alloc() alignment = 64 : memref<16x16xi8>
  %output = memref.alloc() alignment = 64 : memref<16x16xi8>

  scf.for %i = %c0 to %n step %c1 {
    scf.for %j = %c0 to %n step %c1 {
      %ii = arith.index_cast %i : index to i32
      %jj = arith.index_cast %j : index to i32
      %mul = arith.muli %ii, %c16_i32 : i32
      %val = arith.addi %mul, %jj : i32
      %v8 = arith.trunci %val : i32 to i8
      memref.store %v8, %input[%i, %j] : memref<16x16xi8>
    }
  }

  buckyball.mset <{bank = 0 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 0 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.custom "transpose_i8" <{sources = array<i64: 0>, destination = 1 : i64}> {rows = 16 : i64, cols = 16 : i64}
  buckyball.mvout %output <{bank = 1 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.mset <{bank = 0 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>

  func.call @check_result(%output) : (memref<16x16xi8>) -> ()
  memref.dealloc %input : memref<16x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero_i8 : i8
}
