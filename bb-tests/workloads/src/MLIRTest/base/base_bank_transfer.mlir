func.func private @check_result(memref<2x48xi8>) -> ()

// The groups of a two-column bank appended to a one-column bank.
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %first = memref.alloc() alignment = 64 : memref<2x16xi8>
  %second = memref.alloc() alignment = 64 : memref<2x32xi8>
  %result = memref.alloc() alignment = 64 : memref<2x48xi8>

  scf.for %row = %c0 to %c2 step %c1 {
    scf.for %column = %c0 to %c32 step %c1 {
      %r = arith.index_cast %row : index to i32
      %c = arith.index_cast %column : index to i32
      %c32_i32 = arith.constant 32 : i32
      %offset = arith.muli %r, %c32_i32 : i32
      %value = arith.addi %offset, %c : i32
      %b = arith.trunci %value : i32 to i8
      %high = arith.constant 64 : i8
      %second_value = arith.addi %b, %high : i8
      memref.store %second_value, %second[%row, %column] : memref<2x32xi8>
      %in_first = arith.cmpi ult, %column, %c16 : index
      scf.if %in_first {
        %c16_i32 = arith.constant 16 : i32
        %first_offset = arith.muli %r, %c16_i32 : i32
        %first_value = arith.addi %first_offset, %c : i32
        %a = arith.trunci %first_value : i32 to i8
        memref.store %a, %first[%row, %column] : memref<2x16xi8>
      }
    }
  }

  %a = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = buckyball.bank_alloc <{row = 1 : i64, col = 2 : i64}>
  %la = buckyball.bank_mvin %first, %a <{depth = 2 : i64, stride = 1 : i64}> : memref<2x16xi8>
  %lb = buckyball.bank_mvin %second, %b <{depth = 2 : i64, stride = 1 : i64}> : memref<2x32xi8>
  %merged = buckyball.bank_transfer %lb, %la
  %stored = buckyball.bank_mvout %result, %merged <{depth = 2 : i64, stride = 1 : i64}> : memref<2x48xi8>
  buckyball.bank_release %stored

  func.call @check_result(%result) : (memref<2x48xi8>) -> ()
  memref.dealloc %first : memref<2x16xi8>
  memref.dealloc %second : memref<2x32xi8>
  memref.dealloc %result : memref<2x48xi8>
  return %zero_i8 : i8
}
