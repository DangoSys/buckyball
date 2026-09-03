func.func private @check_result(memref<12x16xi8>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %sixteen = arith.constant 16 : index
  %twenty = arith.constant 20 : index
  %eight = arith.constant 8 : index
  %depth16 = arith.constant 16 : i64
  %depth4 = arith.constant 4 : i64
  %depth12 = arith.constant 12 : i64
  %stride = arith.constant 1 : i64
  %input = memref.alloc() alignment = 64 : memref<16x4xi32>
  %scales = memref.alloc() alignment = 64 : memref<4x4xf32>
  %output_seed = memref.alloc() alignment = 64 : memref<12x16xi8>
  %output = memref.alloc() alignment = 64 : memref<12x16xi8>
  linalg.fill ins(%one_f32 : f32) outs(%scales : memref<4x4xf32>)
  linalg.fill ins(%zero_i8 : i8) outs(%output_seed : memref<12x16xi8>)
  linalg.fill ins(%zero_i8 : i8) outs(%output : memref<12x16xi8>)
  scf.for %position = %zero to %four step %one {
    scf.for %channel = %zero to %sixteen step %one {
      %position_base = arith.muli %position, %twenty : index
      %with_channel = arith.addi %position_base, %channel : index
      %value = arith.subi %with_channel, %eight : index
      %value_i32 = arith.index_cast %value : index to i32
      %channel_row = arith.divui %channel, %four : index
      %row_base = arith.muli %position, %four : index
      %row = arith.addi %row_base, %channel_row : index
      %lane = arith.remui %channel, %four : index
      memref.store %value_i32, %input[%row, %lane] : memref<16x4xi32>
    }
  }
  %input_bank = buckyball.bank_alloc
  %scale_bank = buckyball.bank_alloc
  %output_bank = buckyball.bank_alloc
  %loaded_input = buckyball.bank_mvin %input %input_bank %depth16 %stride
      : memref<16x4xi32> i64 i64 i64
  %loaded_scales = buckyball.bank_mvin %scales %scale_bank %depth4 %stride
      : memref<4x4xf32> i64 i64 i64
  %loaded_output = buckyball.bank_mvin %output_seed %output_bank %depth12 %stride
      : memref<12x16xi8> i64 i64 i64
  buckyball.quant_i32_to_i8
      %loaded_input, %loaded_scales, %loaded_output, %depth16
      {outputBase = 3 : i64, outputHeight = 2 : i64,
       outputStride = 4 : i64, outputWidth = 2 : i64, relu = true} : i64
  %stored = buckyball.bank_mvout %output %loaded_output %depth12 %stride
      : memref<12x16xi8> i64 i64 i64
  buckyball.fence
  func.call @check_result(%output) : (memref<12x16xi8>) -> ()
  buckyball.bank_release %loaded_input : i64
  buckyball.bank_release %loaded_scales : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %input : memref<16x4xi32>
  memref.dealloc %scales : memref<4x4xf32>
  memref.dealloc %output_seed : memref<12x16xi8>
  memref.dealloc %output : memref<12x16xi8>
  return %zero_i8 : i8
}
