func.func private @check_result(memref<16x4xi32>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %four_i32 = arith.constant 4 : i32
  %sixteen_i32 = arith.constant 16 : i32
  %iter = arith.constant 16 : i64
  %group = arith.constant 0 : i64
  %stride = arith.constant 1 : i64
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
  %bank = buckyball.bank_alloc {col = 1 : i64}
  %loaded = buckyball.bank_mvin %input %bank %iter %stride : memref<16x4xi32> i64 i64 i64
  buckyball.relu %loaded, %group, %iter, %iter : i64
  %stored = buckyball.bank_mvout %output %loaded %iter %stride : memref<16x4xi32> i64 i64 i64
  buckyball.fence
  func.call @check_result(%output) : (memref<16x4xi32>) -> ()
  buckyball.bank_release %stored : i64
  memref.dealloc %input : memref<16x4xi32>
  memref.dealloc %output : memref<16x4xi32>
  return %zero_i8 : i8
}
