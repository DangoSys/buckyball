func.func private @check_result(memref<4x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %sixteen_i32 = arith.constant 16 : i32
  %input = memref.alloc() : memref<4x16xi8>
  %table = memref.alloc() : memref<16x16xi8>
  %output = memref.alloc() : memref<4x16xi8>
  scf.for %row = %zero to %sixteen step %one {
    scf.for %lane = %zero to %sixteen step %one {
      %row32 = arith.index_cast %row : index to i32
      %lane32 = arith.index_cast %lane : index to i32
      %base = arith.muli %row32, %sixteen_i32 : i32
      %index = arith.addi %base, %lane32 : i32
      %value = arith.trunci %index : i32 to i8
      memref.store %value, %table[%row, %lane] : memref<16x16xi8>
    }
  }
  scf.for %row = %zero to %four step %one {
    scf.for %lane = %zero to %sixteen step %one {
      %row32 = arith.index_cast %row : index to i32
      %lane32 = arith.index_cast %lane : index to i32
      %base = arith.muli %row32, %sixteen_i32 : i32
      %index = arith.addi %base, %lane32 : i32
      %value = arith.trunci %index : i32 to i8
      memref.store %value, %input[%row, %lane] : memref<4x16xi8>
    }
  }
  buckyball.mset <{bank = 0 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 0 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x16xi8>
  buckyball.mvin %table <{bank = 1 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.custom "lookup_i8" <{sources = array<i64: 0, 1>, destination = 2 : i64}> {rows = 4 : i64}
  buckyball.mvout %output <{bank = 2 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x16xi8>
  buckyball.mset <{bank = 0 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%output) : (memref<4x16xi8>) -> ()
  memref.dealloc %input : memref<4x16xi8>
  memref.dealloc %table : memref<16x16xi8>
  memref.dealloc %output : memref<4x16xi8>
  return %zero_i8 : i8
}
