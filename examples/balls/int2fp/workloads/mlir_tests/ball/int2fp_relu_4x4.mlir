func.func private @check_result(memref<4x4xf32>) -> ()

memref.global "private" constant @input : memref<4x4xi32> = dense<[
  [-8, -1, 0, 1], [2, -3, 4, -5],
  [6, 7, -8, 9], [-10, 11, 12, -13]
]>
memref.global "private" constant @scales : memref<4x4xf32> = dense<1.0>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %depth = arith.constant 4 : i64
  %stride = arith.constant 1 : i64
  %input = memref.get_global @input : memref<4x4xi32>
  %scales = memref.get_global @scales : memref<4x4xf32>
  %output = memref.alloc() alignment = 64 : memref<4x4xf32>
  %input_bank = buckyball.bank_alloc
  %scale_bank = buckyball.bank_alloc
  %output_bank = buckyball.bank_alloc
  %loaded_input = buckyball.bank_mvin %input %input_bank %depth %stride
      : memref<4x4xi32> i64 i64 i64
  %loaded_scales = buckyball.bank_mvin %scales %scale_bank %depth %stride
      : memref<4x4xf32> i64 i64 i64
  buckyball.int32_to_fp32
      %loaded_input, %loaded_scales, %output_bank, %depth {relu = true} : i64
  %stored = buckyball.bank_mvout %output %output_bank %depth %stride
      : memref<4x4xf32> i64 i64 i64
  buckyball.fence
  func.call @check_result(%output) : (memref<4x4xf32>) -> ()
  buckyball.bank_release %loaded_input : i64
  buckyball.bank_release %loaded_scales : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %output : memref<4x4xf32>
  return %zero_i8 : i8
}
