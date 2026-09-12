func.func private @check_result(memref<16x4xi32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %depth = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %a = memref.alloc() : memref<16x4xi32>
  %b = memref.alloc() : memref<16x4xi32>
  %out = memref.alloc() : memref<16x4xi32>
  scf.for %row = %zero to %sixteen step %one {
    scf.for %lane = %zero to %four step %one {
      %r = arith.index_cast %row : index to i32
      %l = arith.index_cast %lane : index to i32
      %x = arith.addi %r, %l : i32
      %y = arith.subi %l, %r : i32
      memref.store %x, %a[%row, %lane] : memref<16x4xi32>
      memref.store %y, %b[%row, %lane] : memref<16x4xi32>
    }
  }
  %a_bank = buckyball.bank_alloc
  %b_bank = buckyball.bank_alloc
  %out_bank = buckyball.bank_alloc
  %a_loaded = buckyball.bank_mvin %a %a_bank %depth %stride
      : memref<16x4xi32> i64 i64 i64
  %b_loaded = buckyball.bank_mvin %b %b_bank %depth %stride
      : memref<16x4xi32> i64 i64 i64
  buckyball.matadd %a_loaded, %b_loaded, %out_bank, %depth : i64
  %stored = buckyball.bank_mvout %out %out_bank %depth %stride
      : memref<16x4xi32> i64 i64 i64
  buckyball.fence
  func.call @check_result(%out) : (memref<16x4xi32>) -> ()
  buckyball.bank_release %a_loaded : i64
  buckyball.bank_release %b_loaded : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %a : memref<16x4xi32>
  memref.dealloc %b : memref<16x4xi32>
  memref.dealloc %out : memref<16x4xi32>
  return %zero_i8 : i8
}
