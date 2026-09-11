func.func private @check_result(memref<16x16xi32>) -> ()
func.func @main() -> i8 {
  %z8 = arith.constant 0 : i8
  %z32 = arith.constant 0 : i32
  %z = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %seven = arith.constant 7 : index
  %dim = arith.constant 16 : index
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %mode = arith.constant 0 : i64
  %at = memref.alloc() alignment = 64 : memref<16x16xi8>
  %b = memref.alloc() alignment = 64 : memref<16x16xi8>
  %c = memref.alloc() alignment = 64 : memref<16x16xi32>
  linalg.fill ins(%z32 : i32) outs(%c : memref<16x16xi32>)
  scf.for %row = %z to %dim step %one {
    scf.for %col = %z to %dim step %one {
      %r3 = arith.muli %row, %three : index
      %c5 = arith.muli %col, %five : index
      %asum = arith.addi %r3, %c5 : index
      %arem = arith.remui %asum, %seven : index
      %aval = arith.subi %arem, %three : index
      %atv = arith.index_cast %aval : index to i8
      %r2 = arith.muli %row, %two : index
      %c3 = arith.muli %col, %three : index
      %bsum = arith.addi %r2, %c3 : index
      %brem = arith.remui %bsum, %five : index
      %bval = arith.subi %brem, %two : index
      %bv = arith.index_cast %bval : index to i8
      memref.store %atv, %at[%row, %col] : memref<16x16xi8>
      memref.store %bv, %b[%row, %col] : memref<16x16xi8>
    }
  }
  %ab = buckyball.bank_alloc
  %bb = buckyball.bank_alloc
  %cb = buckyball.bank_alloc {col = 4 : i64}
  %al = buckyball.bank_mvin %at %ab %depth %stride
      : memref<16x16xi8> i64 i64 i64
  %bl = buckyball.bank_mvin %b %bb %depth %stride
      : memref<16x16xi8> i64 i64 i64
  %cl = buckyball.bank_mvin %c %cb %depth %stride
      : memref<16x16xi32> i64 i64 i64
  buckyball.vecmat16 %al, %bl, %cl, %depth, %mode : i64
  %stored = buckyball.bank_mvout %c %cl %depth %stride
      : memref<16x16xi32> i64 i64 i64
  buckyball.fence
  func.call @check_result(%c) : (memref<16x16xi32>) -> ()
  buckyball.bank_release %al : i64
  buckyball.bank_release %bl : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %at : memref<16x16xi8>
  memref.dealloc %b : memref<16x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %z8 : i8
}
