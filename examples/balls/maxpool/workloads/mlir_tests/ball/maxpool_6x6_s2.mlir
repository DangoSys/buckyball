func.func private @check_result(memref<9x16xi8>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %thirty_six = arith.constant 36 : index
  %sixteen = arith.constant 16 : index
  %depth36 = arith.constant 36 : i64
  %depth9 = arith.constant 9 : i64
  %zero64 = arith.constant 0 : i64
  %three64 = arith.constant 3 : i64
  %stride = arith.constant 1 : i64
  %input = memref.alloc() : memref<36x16xi8>
  %output = memref.alloc() : memref<9x16xi8>
  scf.for %row = %zero to %thirty_six step %one {
    %value = arith.index_cast %row : index to i8
    scf.for %lane = %zero to %sixteen step %one {
      memref.store %value, %input[%row, %lane] : memref<36x16xi8>
    }
  }
  %in_bank = buckyball.bank_alloc
  %out_bank = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %in_bank %depth36 %stride : memref<36x16xi8> i64 i64 i64
  buckyball.maxpool %loaded, %out_bank, %depth9, %zero64, %zero64, %three64 {inputSide = 6 : i64, outputSide = 3 : i64, kernel = 2 : i64, stride = 2 : i64, padding = 0 : i64, startRow = 0 : i64, startCol = 0 : i64} : i64 i64 i64 i64 i64 i64
  %stored = buckyball.bank_mvout %output %out_bank %depth9 %stride : memref<9x16xi8> i64 i64 i64
  buckyball.fence
  func.call @check_result(%output) : (memref<9x16xi8>) -> ()
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %input : memref<36x16xi8>
  memref.dealloc %output : memref<9x16xi8>
  return %zero_i8 : i8
}
