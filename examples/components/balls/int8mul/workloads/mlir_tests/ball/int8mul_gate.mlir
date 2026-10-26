func.func private @check_result(memref<4x16xi8>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_i8 = arith.constant 1 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %ratio = arith.constant 1.0 : f32
  %gate = memref.alloc() : memref<1x16xi8>
  %input = memref.alloc() : memref<4x16xi8>
  %output = memref.alloc() : memref<4x16xi8>
  scf.for %lane = %zero to %sixteen step %one {
    memref.store %one_i8, %gate[%zero, %lane] : memref<1x16xi8>
  }
  scf.for %row = %zero to %four step %one {
    scf.for %lane = %zero to %sixteen step %one {
      %value = arith.index_cast %lane : index to i8
      memref.store %value, %input[%row, %lane] : memref<4x16xi8>
    }
  }
  buckyball.mset <{bank = 0 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %gate <{bank = 0 : i64, depth = 1 : i64, stride = 1 : i64}> : memref<1x16xi8>
  buckyball.mvin %input <{bank = 1 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x16xi8>
  buckyball.custom "gate_mul_i8" args(%ratio : f32) <{sources = array<i64: 0, 1>, destination = 2 : i64}> {rows = 4 : i64, gate_row = 0 : i64}
  buckyball.mvout %output <{bank = 2 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x16xi8>
  func.call @check_result(%output) : (memref<4x16xi8>) -> ()
  buckyball.mset <{bank = 0 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  memref.dealloc %gate : memref<1x16xi8>
  memref.dealloc %input : memref<4x16xi8>
  memref.dealloc %output : memref<4x16xi8>
  return %zero_i8 : i8
}
