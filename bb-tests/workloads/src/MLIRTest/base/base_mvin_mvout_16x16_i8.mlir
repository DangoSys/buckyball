func.func private @check_result(memref<16x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %c16 = arith.constant 16 : i32
  %bank = arith.constant 0 : i64
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %input = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>
  %output = memref.alloc() {alignment = 64 : i64} : memref<16x16xi8>

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

  buckyball.mset %bank {row = 1 : i64, col = 1 : i64} : i64
  buckyball.mvin %input %bank %depth %stride : memref<16x16xi8> i64 i64 i64
  buckyball.mvout %output %bank %depth %stride : memref<16x16xi8> i64 i64 i64
  buckyball.fence
  buckyball.mset %bank {alloc = false, row = 0 : i64, col = 0 : i64} : i64

  func.call @check_result(%output) : (memref<16x16xi8>) -> ()
  memref.dealloc %input : memref<16x16xi8>
  memref.dealloc %output : memref<16x16xi8>
  return %zero_i8 : i8
}
