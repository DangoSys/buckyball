func.func private @check_result(memref<64x4xi32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_i32 = arith.constant 0 : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %four = arith.constant 4 : index
  %eight = arith.constant 8 : index
  %sixteen = arith.constant 16 : index
  %depth4 = arith.constant 4 : i64
  %depth16 = arith.constant 16 : i64
  %depth64 = arith.constant 64 : i64
  %stride = arith.constant 1 : i64
  %config = arith.constant 268501008 : i64
  %true = arith.constant true
  %false = arith.constant false
  %zero64 = arith.constant 0 : i64

  %identity = memref.alloc() alignment = 64 : memref<16x16xi8>
  %weight0 = memref.alloc() alignment = 64 : memref<16x16xi8>
  %weight1 = memref.alloc() alignment = 64 : memref<16x16xi8>
  %bias = memref.alloc() alignment = 64 : memref<4x4xi32>
  %output = memref.alloc() alignment = 64 : memref<64x4xi32>
  linalg.fill ins(%zero_i32 : i32) outs(%output : memref<64x4xi32>)

  scf.for %row = %zero to %sixteen step %one {
    scf.for %column = %zero to %sixteen step %one {
      %same = arith.cmpi eq, %row, %column : index
      %identity_i32 = arith.select %same, %one, %zero : index
      %identity_i8 = arith.index_cast %identity_i32 : index to i8
      %row_i32 = arith.index_cast %row : index to i32
      %column_i32 = arith.index_cast %column : index to i32
      %sum = arith.addi %row_i32, %column_i32 : i32
      %difference = arith.subi %row_i32, %column_i32 : i32
      %sum_i8 = arith.trunci %sum : i32 to i8
      %difference_i8 = arith.trunci %difference : i32 to i8
      memref.store %identity_i8, %identity[%row, %column] : memref<16x16xi8>
      memref.store %sum_i8, %weight0[%row, %column] : memref<16x16xi8>
      memref.store %difference_i8, %weight1[%row, %column] : memref<16x16xi8>
    }
    %bias_value = arith.subi %row, %eight : index
    %bias_i32 = arith.index_cast %bias_value : index to i32
    %bias_row = arith.divui %row, %four : index
    %bias_lane = arith.remui %row, %four : index
    memref.store %bias_i32, %bias[%bias_row, %bias_lane] : memref<4x4xi32>
  }

  %bias_bank = buckyball.bank_alloc
  %a0_bank = buckyball.bank_alloc
  %b0_bank = buckyball.bank_alloc
  %a1_bank = buckyball.bank_alloc
  %b1_bank = buckyball.bank_alloc
  %result_bank = buckyball.bank_alloc
  %loaded_bias = buckyball.bank_mvin %bias %bias_bank %depth4 %stride
      : memref<4x4xi32> i64 i64 i64
  %loaded_a0 = buckyball.bank_mvin %identity %a0_bank %depth16 %stride
      : memref<16x16xi8> i64 i64 i64
  %loaded_b0 = buckyball.bank_mvin %weight0 %b0_bank %depth16 %stride
      : memref<16x16xi8> i64 i64 i64
  %loaded_a1 = buckyball.bank_mvin %identity %a1_bank %depth16 %stride
      : memref<16x16xi8> i64 i64 i64
  %loaded_b1 = buckyball.bank_mvin %weight1 %b1_bank %depth16 %stride
      : memref<16x16xi8> i64 i64 i64
  %bias_state = buckyball.bank_smatmul_bias %loaded_bias %zero64 : i64 i64
  %partial = buckyball.bank_smatmul %loaded_a0 %loaded_b0 %result_bank %config %true %false %zero64 : i64 i64 i64 i64
  %result = buckyball.bank_smatmul %loaded_a1 %loaded_b1 %partial %config %false %true %zero64 : i64 i64 i64 i64
  %stored = buckyball.bank_mvout %output %result %depth64 %stride
      : memref<64x4xi32> i64 i64 i64
  buckyball.fence

  func.call @check_result(%output) : (memref<64x4xi32>) -> ()
  buckyball.bank_release %bias_state : i64
  buckyball.bank_release %loaded_a0 : i64
  buckyball.bank_release %loaded_b0 : i64
  buckyball.bank_release %loaded_a1 : i64
  buckyball.bank_release %loaded_b1 : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %identity : memref<16x16xi8>
  memref.dealloc %weight0 : memref<16x16xi8>
  memref.dealloc %weight1 : memref<16x16xi8>
  memref.dealloc %bias : memref<4x4xi32>
  memref.dealloc %output : memref<64x4xi32>
  return %zero_i8 : i8
}
