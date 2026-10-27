func.func private @check_result(memref<16x16xi32>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %initial = arith.constant 7 : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %seven = arith.constant 7 : index
  %sixteen = arith.constant 16 : index
  %at = memref.alloc() : memref<16x16xi8>
  %b = memref.alloc() : memref<16x16xi8>
  %c = memref.alloc() : memref<16x16xi32>
  linalg.fill ins(%initial : i32) outs(%c : memref<16x16xi32>)
  scf.for %k = %zero to %sixteen step %one {
    scf.for %lane = %zero to %sixteen step %one {
      %k3 = arith.muli %k, %three : index
      %l5 = arith.muli %lane, %five : index
      %asum = arith.addi %k3, %l5 : index
      %arem = arith.remui %asum, %seven : index
      %aval = arith.subi %arem, %three : index
      %av = arith.index_cast %aval : index to i8
      %k2 = arith.muli %k, %two : index
      %l3 = arith.muli %lane, %three : index
      %bsum = arith.addi %k2, %l3 : index
      %brem = arith.remui %bsum, %five : index
      %bval = arith.subi %brem, %two : index
      %bv = arith.index_cast %bval : index to i8
      memref.store %av, %at[%k, %lane] : memref<16x16xi8>
      memref.store %bv, %b[%k, %lane] : memref<16x16xi8>
    }
  }
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = true, row = 1 : i64, col = 4 : i64}>
  buckyball.mvin %at <{bank = 1 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.mvin %b <{bank = 2 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi8>
  buckyball.mvin %c <{bank = 3 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi32>
  // C += A^T * B with K = 16.
  vector_ball.matmul_at_i8 <{at = 1 : i64, b = 2 : i64, c = 3 : i64, rows = 16 : i64}>
  buckyball.mvout %c <{bank = 3 : i64, depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi32>
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  func.call @check_result(%c) : (memref<16x16xi32>) -> ()
  memref.dealloc %at : memref<16x16xi8>
  memref.dealloc %b : memref<16x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %zero_i8 : i8
}
