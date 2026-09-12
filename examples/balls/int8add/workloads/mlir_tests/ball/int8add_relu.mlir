func.func private @check_result(memref<4x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %depth = arith.constant 4 : i64
  %stride = arith.constant 1 : i64
  %ratio = arith.constant 1.0 : f32
  %a = memref.alloc() : memref<4x16xi8>
  %b = memref.alloc() : memref<4x16xi8>
  %out = memref.alloc() : memref<4x16xi8>
  scf.for %row = %zero to %four step %one {
    scf.for %lane = %zero to %sixteen step %one {
      %r = arith.index_cast %row : index to i32
      %l = arith.index_cast %lane : index to i32
      %a32 = arith.subi %l, %r : i32
      %b32 = arith.subi %r, %l : i32
      %a8 = arith.trunci %a32 : i32 to i8
      %b8 = arith.trunci %b32 : i32 to i8
      memref.store %a8, %a[%row, %lane] : memref<4x16xi8>
      memref.store %b8, %b[%row, %lane] : memref<4x16xi8>
    }
  }
  %a_bank = buckyball.bank_alloc
  %b_bank = buckyball.bank_alloc
  %out_bank = buckyball.bank_alloc
  %a_loaded = buckyball.bank_mvin %a %a_bank %depth %stride : memref<4x16xi8> i64 i64 i64
  %b_loaded = buckyball.bank_mvin %b %b_bank %depth %stride : memref<4x16xi8> i64 i64 i64
  buckyball.int8add %a_loaded, %b_loaded, %out_bank, %depth, %ratio, %ratio {relu = true} : i64
  %stored = buckyball.bank_mvout %out %out_bank %depth %stride : memref<4x16xi8> i64 i64 i64
  buckyball.fence
  func.call @check_result(%out) : (memref<4x16xi8>) -> ()
  buckyball.bank_release %a_loaded : i64
  buckyball.bank_release %b_loaded : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %a : memref<4x16xi8>
  memref.dealloc %b : memref<4x16xi8>
  memref.dealloc %out : memref<4x16xi8>
  return %zero_i8 : i8
}
