func.func private @check_result(memref<16x16xi32>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_i32 = arith.constant 0 : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %seven = arith.constant 7 : index
  %sixteen = arith.constant 16 : index
  %base = arith.constant 0 : i64
  %a = memref.alloc() : memref<16x16xi8>
  %b = memref.alloc() : memref<16x16xi8>
  %c = memref.alloc() : memref<16x16xi32>
  linalg.fill ins(%zero_i32 : i32) outs(%c : memref<16x16xi32>)
  scf.for %row = %zero to %sixteen step %one {
    scf.for %col = %zero to %sixteen step %one {
      %r3 = arith.muli %row, %three : index
      %c5 = arith.muli %col, %five : index
      %asum = arith.addi %r3, %c5 : index
      %arem = arith.remui %asum, %seven : index
      %aval = arith.subi %arem, %three : index
      %av = arith.index_cast %aval : index to i8
      %r2 = arith.muli %row, %two : index
      %c3 = arith.muli %col, %three : index
      %bsum = arith.addi %r2, %c3 : index
      %brem = arith.remui %bsum, %five : index
      %bval = arith.subi %brem, %two : index
      %bv = arith.index_cast %bval : index to i8
      memref.store %av, %a[%row, %col] : memref<16x16xi8>
      memref.store %bv, %b[%row, %col] : memref<16x16xi8>
    }
  }
  buckyball.mset <{bank = 0 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = true, row = 1 : i64, col = 4 : i64}>
  buckyball.mvin %a <{bank = 0 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.mvin %b <{bank = 1 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  // C = A @ B on the output-stationary array.
  buckyball.custom "matmul_i8_tile" args(%base, %base, %base : i64, i64, i64) <{sources = array<i64: 0, 1>, destination = 3 : i64}>
  buckyball.mvout %c <{bank = 3 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi32>
  func.call @check_result(%c) : (memref<16x16xi32>) -> ()
  buckyball.mset <{bank = 0 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  memref.dealloc %a : memref<16x16xi8>
  memref.dealloc %b : memref<16x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %zero_i8 : i8
}
