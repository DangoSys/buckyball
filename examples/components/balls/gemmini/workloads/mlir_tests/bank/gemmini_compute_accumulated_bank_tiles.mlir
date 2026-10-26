func.func private @check_result(memref<16x16xi32>) -> ()

// Each group holds a whole bank of K: A as blocks of sixteen rows, B as one
// row per K. A[i, kk] = (3i + 5kk) % 7 - 3 with kk = first + 16 * (q / 16) + col
// at bank row q, i = q % 16.
func.func @fill_a(%a: memref<1024x16xi8>, %first: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c16 = arith.constant 16 : index
  %lines = arith.constant 1024 : index
  scf.for %q = %c0 to %lines step %c1 {
    %i = arith.remui %q, %c16 : index
    %block = arith.subi %q, %i : index
    %start = arith.addi %first, %block : index
    scf.for %col = %c0 to %c16 step %c1 {
      %kk = arith.addi %start, %col : index
      %i3 = arith.muli %i, %c3 : index
      %k5 = arith.muli %kk, %c5 : index
      %sum = arith.addi %i3, %k5 : index
      %rem = arith.remui %sum, %c7 : index
      %value = arith.subi %rem, %c3 : index
      %v8 = arith.index_cast %value : index to i8
      memref.store %v8, %a[%q, %col] : memref<1024x16xi8>
    }
  }
  return
}

// B[kk, j] = (2kk + 3j) % 5 - 2 with kk = first + q at bank row q.
func.func @fill_b(%b: memref<1024x16xi8>, %first: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c16 = arith.constant 16 : index
  %lines = arith.constant 1024 : index
  scf.for %q = %c0 to %lines step %c1 {
    %kk = arith.addi %first, %q : index
    scf.for %j = %c0 to %c16 step %c1 {
      %k2 = arith.muli %kk, %c2 : index
      %j3 = arith.muli %j, %c3 : index
      %sum = arith.addi %k2, %j3 : index
      %rem = arith.remui %sum, %c5 : index
      %value = arith.subi %rem, %c2 : index
      %v8 = arith.index_cast %value : index to i8
      memref.store %v8, %b[%q, %j] : memref<1024x16xi8>
    }
  }
  return
}

// C = in_shift(7 + A[16 x 2048] @ B[2048 x 16], 2), A and B each gathered
// from two full 1024-row banks.
func.func @main() -> i8 {
  %z8 = arith.constant 0 : i8
  %initial = arith.constant 7 : i32
  %c0 = arith.constant 0 : index
  %lines = arith.constant 1024 : index
  %a0 = memref.alloc() : memref<1024x16xi8>
  %a1 = memref.alloc() : memref<1024x16xi8>
  %b0 = memref.alloc() : memref<1024x16xi8>
  %b1 = memref.alloc() : memref<1024x16xi8>
  %c = memref.alloc() : memref<16x16xi32>
  linalg.fill ins(%initial : i32) outs(%c : memref<16x16xi32>)
  func.call @fill_a(%a0, %c0) : (memref<1024x16xi8>, index) -> ()
  func.call @fill_a(%a1, %lines) : (memref<1024x16xi8>, index) -> ()
  func.call @fill_b(%b0, %c0) : (memref<1024x16xi8>, index) -> ()
  func.call @fill_b(%b1, %lines) : (memref<1024x16xi8>, index) -> ()
  %ab0 = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %bb0 = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %ab1 = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %bb1 = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %cb = buckyball.bank_alloc <{row = 1 : i64, col = 4 : i64}>
  %al0 = buckyball.bank_mvin %a0, %ab0 <{depth = 1024 : i64, stride = 1 : i64}> : memref<1024x16xi8>
  %bl0 = buckyball.bank_mvin %b0, %bb0 <{depth = 1024 : i64, stride = 1 : i64}> : memref<1024x16xi8>
  %al1 = buckyball.bank_mvin %a1, %ab1 <{depth = 1024 : i64, stride = 1 : i64}> : memref<1024x16xi8>
  %bl1 = buckyball.bank_mvin %b1, %bb1 <{depth = 1024 : i64, stride = 1 : i64}> : memref<1024x16xi8>
  %cl = buckyball.bank_mvin %c, %cb <{depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi32>
  %al = buckyball.bank_transfer %al1, %al0
  %bl = buckyball.bank_transfer %bl1, %bl0
  %r = buckyball.bank_custom "matmul_i8_tile" (%al, %bl) -> %cl <{read_destination}> {k = 2048 : i64, shift = 2 : i64}
  %stored = buckyball.bank_mvout %c, %r <{depth = 16 : i64, stride = 1 : i64}> : memref<16x16xi32>
  func.call @check_result(%c) : (memref<16x16xi32>) -> ()
  buckyball.bank_release %al
  buckyball.bank_release %bl
  buckyball.bank_release %stored
  memref.dealloc %a0 : memref<1024x16xi8>
  memref.dealloc %a1 : memref<1024x16xi8>
  memref.dealloc %b0 : memref<1024x16xi8>
  memref.dealloc %b1 : memref<1024x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %z8 : i8
}
