func.func private @check_result(memref<16x16xi32>) -> ()

// A holds K = 96 as six blocks of sixteen rows:
// A[i, kk] = (3i + 5kk) % 7 - 3 with kk = 16 * (q / 16) + col at bank row q,
// i = q % 16.
func.func @fill_a(%a: memref<96x16xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c16 = arith.constant 16 : index
  %c96 = arith.constant 96 : index
  scf.for %q = %c0 to %c96 step %c1 {
    %i = arith.remui %q, %c16 : index
    %block = arith.subi %q, %i : index
    scf.for %col = %c0 to %c16 step %c1 {
      %kk = arith.addi %block, %col : index
      %i3 = arith.muli %i, %c3 : index
      %k5 = arith.muli %kk, %c5 : index
      %sum = arith.addi %i3, %k5 : index
      %rem = arith.remui %sum, %c7 : index
      %value = arith.subi %rem, %c3 : index
      %v8 = arith.index_cast %value : index to i8
      memref.store %v8, %a[%q, %col] : memref<96x16xi8>
    }
  }
  return
}

// B[kk, j] = (2kk + 3j) % 5 - 2 with kk = q at bank row q.
func.func @fill_b(%b: memref<96x16xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c16 = arith.constant 16 : index
  %c96 = arith.constant 96 : index
  scf.for %q = %c0 to %c96 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %k2 = arith.muli %q, %c2 : index
      %j3 = arith.muli %j, %c3 : index
      %sum = arith.addi %k2, %j3 : index
      %rem = arith.remui %sum, %c5 : index
      %value = arith.subi %rem, %c2 : index
      %v8 = arith.index_cast %value : index to i8
      memref.store %v8, %b[%q, %j] : memref<96x16xi8>
    }
  }
  return
}

// C = A[16 x 96] @ B[96 x 16] in one operation of k = 96.
func.func @main() -> i8 {
  %z8 = arith.constant 0 : i8
  %a = memref.alloc() : memref<96x16xi8>
  %b = memref.alloc() : memref<96x16xi8>
  %c = memref.alloc() : memref<16x16xi32>
  func.call @fill_a(%a) : (memref<96x16xi8>) -> ()
  func.call @fill_b(%b) : (memref<96x16xi8>) -> ()
  %ab = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %bb = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %cb = buckyball.bank_alloc <{row = 1 : i64, col = 4 : i64}>
  %al = buckyball.bank_mvin %a, %ab <{depth = 96 : i64, stride = 1 : i64}> : memref<96x16xi8>
  %bl = buckyball.bank_mvin %b, %bb <{depth = 96 : i64, stride = 1 : i64}> : memref<96x16xi8>
  %r = gemmini_ball.bank_matmul_i8_tile %al, %bl -> %cb <{k = 96 : i64}>
  %stored = buckyball.bank_mvout %c, %r <{depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi32>
  func.call @check_result(%c) : (memref<16x16xi32>) -> ()
  buckyball.bank_release %al
  buckyball.bank_release %bl
  buckyball.bank_release %stored
  memref.dealloc %a : memref<96x16xi8>
  memref.dealloc %b : memref<96x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %z8 : i8
}
