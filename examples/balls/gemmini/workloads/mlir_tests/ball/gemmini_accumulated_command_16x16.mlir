func.func private @check_result(memref<16x16xi32>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_i8 = arith.constant 1 : i8
  %initial = arith.constant 7 : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %seven = arith.constant 7 : index
  %sixteen = arith.constant 16 : index
  %depth16 = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %a = memref.alloc() : memref<16x16xi8>
  %b = memref.alloc() : memref<16x16xi8>
  %c = memref.alloc() : memref<16x16xi32>
  linalg.fill ins(%initial : i32) outs(%c : memref<16x16xi32>)
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
  %a_bank = arith.constant 0 : i64
  %b_bank = arith.constant 1 : i64
  %c_bank = arith.constant 3 : i64
  buckyball.mset %a_bank <row = 1, col = 1> : i64
  buckyball.mset %b_bank <row = 1, col = 1> : i64
  buckyball.mset %c_bank <row = 1, col = 4> : i64
  buckyball.mvin %a %a_bank %depth16 %stride : memref<16x16xi8> i64 i64 i64
  buckyball.mvin %b %b_bank %depth16 %stride : memref<16x16xi8> i64 i64 i64
  buckyball.mvin %c %c_bank %depth16 %stride : memref<16x16xi32> i64 i64 i64
  %config = arith.constant 2064 : i64
  %zero64 = arith.constant 0 : i64
  %shift10 = arith.constant 10 : i64
  %shift20 = arith.constant 20 : i64
  %shift30 = arith.constant 30 : i64
  %iter_bits = arith.shli %depth16, %shift30 : i64
  %c_bits = arith.shli %c_bank, %shift20 : i64
  %b_bits = arith.shli %b_bank, %shift10 : i64
  %compute_rs1 = arith.ori %iter_bits, %b_bits : i64
  %compute_rs1_final = arith.ori %compute_rs1, %c_bits : i64
  %compute_rs2 = arith.constant 3 : i64
  buckyball.gemmini "GEMMINI_CONFIG" %zero64, %config : i64
  buckyball.gemmini "GEMMINI_COMPUTE_ACCUMULATED" %compute_rs1_final, %compute_rs2 : i64
  buckyball.mvout %c %c_bank %depth16 %stride : memref<16x16xi32> i64 i64 i64
  buckyball.fence
  func.call @check_result(%c) : (memref<16x16xi32>) -> ()
  buckyball.mset %a_bank <alloc = false, row = 0, col = 0> : i64
  buckyball.mset %b_bank <alloc = false, row = 0, col = 0> : i64
  buckyball.mset %c_bank <alloc = false, row = 0, col = 0> : i64
  memref.dealloc %a : memref<16x16xi8>
  memref.dealloc %b : memref<16x16xi8>
  memref.dealloc %c : memref<16x16xi32>
  return %zero_i8 : i8
}
