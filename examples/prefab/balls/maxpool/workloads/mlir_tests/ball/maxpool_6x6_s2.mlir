func.func private @check_result(memref<9x16xi8>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %thirty_six = arith.constant 36 : index
  %sixteen = arith.constant 16 : index
  %input = memref.alloc() : memref<36x16xi8>
  %output = memref.alloc() : memref<9x16xi8>
  scf.for %row = %zero to %thirty_six step %one {
    %value = arith.index_cast %row : index to i8
    scf.for %lane = %zero to %sixteen step %one {
      memref.store %value, %input[%row, %lane] : memref<36x16xi8>
    }
  }
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 1 : i64, depth = 36 : i64, stride = 1 : i64}> : memref<36x16xi8>
  maxpool_ball.maxpool_i8 <{input = 1 : i64, output = 2 : i64, input_side = 6 : i64, kernel = 2 : i64, stride = 2 : i64, padding = 0 : i64}>
  buckyball.mvout %output <{bank = 2 : i64, depth = 9 : i64, stride = 1 : i64}> : memref<9x16xi8>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%output) : (memref<9x16xi8>) -> ()
  memref.dealloc %input : memref<36x16xi8>
  memref.dealloc %output : memref<9x16xi8>
  return %zero_i8 : i8
}
