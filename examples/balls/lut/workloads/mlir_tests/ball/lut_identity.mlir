func.func private @check_result(memref<4x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %depth4 = arith.constant 4 : i64
  %depth16 = arith.constant 16 : i64
  %stride = arith.constant 1 : i64
  %sixteen_i32 = arith.constant 16 : i32
  %input = memref.alloc() : memref<4x16xi8>
  %table = memref.alloc() : memref<16x16xi8>
  %output = memref.alloc() : memref<4x16xi8>
  scf.for %row = %zero to %sixteen step %one {
    scf.for %lane = %zero to %sixteen step %one {
      %row32 = arith.index_cast %row : index to i32
      %lane32 = arith.index_cast %lane : index to i32
      %base = arith.muli %row32, %sixteen_i32 : i32
      %index = arith.addi %base, %lane32 : i32
      %value = arith.trunci %index : i32 to i8
      memref.store %value, %table[%row, %lane] : memref<16x16xi8>
    }
  }
  scf.for %row = %zero to %four step %one {
    scf.for %lane = %zero to %sixteen step %one {
      %row32 = arith.index_cast %row : index to i32
      %lane32 = arith.index_cast %lane : index to i32
      %base = arith.muli %row32, %sixteen_i32 : i32
      %index = arith.addi %base, %lane32 : i32
      %value = arith.trunci %index : i32 to i8
      memref.store %value, %input[%row, %lane] : memref<4x16xi8>
    }
  }
  %input_bank = buckyball.bank_alloc
  %table_bank = buckyball.bank_alloc
  %output_bank = buckyball.bank_alloc
  %loaded_input = buckyball.bank_mvin %input %input_bank %depth4 %stride : memref<4x16xi8> i64 i64 i64
  %loaded_table = buckyball.bank_mvin %table %table_bank %depth16 %stride : memref<16x16xi8> i64 i64 i64
  buckyball.lut %loaded_input, %loaded_table, %output_bank, %depth4 : i64
  %stored = buckyball.bank_mvout %output %output_bank %depth4 %stride : memref<4x16xi8> i64 i64 i64
  buckyball.fence
  func.call @check_result(%output) : (memref<4x16xi8>) -> ()
  buckyball.bank_release %loaded_input : i64
  buckyball.bank_release %loaded_table : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %input : memref<4x16xi8>
  memref.dealloc %table : memref<16x16xi8>
  memref.dealloc %output : memref<4x16xi8>
  return %zero_i8 : i8
}
