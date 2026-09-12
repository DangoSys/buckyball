func.func private @check_result(memref<4x16xi8>) -> ()
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_i8 = arith.constant 1 : i8
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %depth4 = arith.constant 4 : i64
  %depth1 = arith.constant 1 : i64
  %stride = arith.constant 1 : i64
  %ratio = arith.constant 1.0 : f32
  %gate_row = arith.constant 0 : i64
  %gate = memref.alloc() : memref<1x16xi8>
  %input = memref.alloc() : memref<4x16xi8>
  %output = memref.alloc() : memref<4x16xi8>
  scf.for %lane = %zero to %sixteen step %one {
    memref.store %one_i8, %gate[%zero, %lane] : memref<1x16xi8>
  }
  scf.for %row = %zero to %four step %one {
    scf.for %lane = %zero to %sixteen step %one {
      %value = arith.index_cast %lane : index to i8
      memref.store %value, %input[%row, %lane] : memref<4x16xi8>
    }
  }
  %gate_bank = buckyball.bank_alloc
  %input_bank = buckyball.bank_alloc
  %output_bank = buckyball.bank_alloc
  %loaded_gate = buckyball.bank_mvin %gate %gate_bank %depth1 %stride : memref<1x16xi8> i64 i64 i64
  %loaded_input = buckyball.bank_mvin %input %input_bank %depth4 %stride : memref<4x16xi8> i64 i64 i64
  buckyball.int8mul %loaded_gate, %loaded_input, %output_bank, %depth4, %ratio, %gate_row : i64
  %stored = buckyball.bank_mvout %output %output_bank %depth4 %stride : memref<4x16xi8> i64 i64 i64
  buckyball.fence
  func.call @check_result(%output) : (memref<4x16xi8>) -> ()
  buckyball.bank_release %loaded_gate : i64
  buckyball.bank_release %loaded_input : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %gate : memref<1x16xi8>
  memref.dealloc %input : memref<4x16xi8>
  memref.dealloc %output : memref<4x16xi8>
  return %zero_i8 : i8
}
