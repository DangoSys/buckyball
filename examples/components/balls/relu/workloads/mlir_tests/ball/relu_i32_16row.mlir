func.func private @check_result(memref<16x4xi32>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %four_i32 = arith.constant 4 : i32
  %sixteen_i32 = arith.constant 16 : i32
  %input = memref.alloc() : memref<16x4xi32>
  %output = memref.alloc() : memref<16x4xi32>
  scf.for %row = %zero to %sixteen step %one {
    scf.for %lane = %zero to %four step %one {
      %r = arith.index_cast %row : index to i32
      %l = arith.index_cast %lane : index to i32
      %base = arith.muli %r, %four_i32 : i32
      %value = arith.addi %base, %l : i32
      %signed = arith.subi %value, %sixteen_i32 : i32
      memref.store %signed, %input[%row, %lane] : memref<16x4xi32>
    }
  }
  buckyball.mset <{bank = 0 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 0 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x4xi32>
  buckyball.custom "relu_i32" <{sources = array<i64: 0>, destination = 0 : i64}> {rows = 16 : i64, stride = 16 : i64, group = 0 : i64}
  buckyball.mvout %output <{bank = 0 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x4xi32>
  buckyball.mset <{bank = 0 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%output) : (memref<16x4xi32>) -> ()
  memref.dealloc %input : memref<16x4xi32>
  memref.dealloc %output : memref<16x4xi32>
  return %zero_i8 : i8
}
