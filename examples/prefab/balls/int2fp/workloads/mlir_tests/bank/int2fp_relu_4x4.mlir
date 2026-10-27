func.func private @check_result(memref<4x4xf32>) -> ()

memref.global "private" constant @input : memref<4x4xi32> = dense<[
  [-8, -1, 0, 1], [2, -3, 4, -5],
  [6, 7, -8, 9], [-10, 11, 12, -13]
]>
memref.global "private" constant @scales : memref<4x4xf32> = dense<1.0>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %input = memref.get_global @input : memref<4x4xi32>
  %scales = memref.get_global @scales : memref<4x4xf32>
  %output = memref.alloc() alignment = 64 : memref<4x4xf32>
  %input_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %scale_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %output_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %loaded_input = buckyball.bank_mvin %input, %input_bank <{depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  %loaded_scales = buckyball.bank_mvin %scales, %scale_bank <{depth = 4 : i64, stride = 1 : i64}> : memref<4x4xf32>
  %result = int2fp_ball.bank_int32_to_fp32 %loaded_input, %loaded_scales -> %output_bank <{rows = 1 : i64, relu}>
  %stored = buckyball.bank_mvout %output, %result <{depth = 4 : i64, stride = 1 : i64}> : memref<4x4xf32>
  func.call @check_result(%output) : (memref<4x4xf32>) -> ()
  buckyball.bank_release %loaded_input
  buckyball.bank_release %loaded_scales
  buckyball.bank_release %stored
  memref.dealloc %output : memref<4x4xf32>
  return %zero_i8 : i8
}
