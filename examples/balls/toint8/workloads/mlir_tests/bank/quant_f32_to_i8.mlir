func.func private @check_result(memref<1x16xi8>) -> ()

memref.global "private" constant @input : memref<4x4xf32> = dense<[
  [-3.000000e+02, -2.550000e+02, -5.000000e+00, -3.000000e+00],
  [-1.000000e+00, 0.000000e+00, 1.000000e+00, 3.000000e+00],
  [5.000000e+00, 2.540000e+02, 2.550000e+02, 3.000000e+02],
  [2.000000e+00, 6.000000e+00, 1.000000e+01, 1.400000e+01]
]>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %depth4 = arith.constant 4 : i64
  %depth1 = arith.constant 1 : i64
  %stride = arith.constant 1 : i64
  %scale = arith.constant 5.000000e-01 : f32
  %input = memref.get_global @input : memref<4x4xf32>
  %output = memref.alloc() alignment = 64 : memref<1x16xi8>
  %input_bank = buckyball.bank_alloc
  %output_bank = buckyball.bank_alloc
  %loaded = buckyball.bank_mvin %input %input_bank %depth4 %stride
      : memref<4x4xf32> i64 i64 i64
  %result = buckyball.bank_quant_f32_to_i8
      %loaded %output_bank %depth4 %scale : i64 i64 i64 f32
  %stored = buckyball.bank_mvout %output %result %depth1 %stride
      : memref<1x16xi8> i64 i64 i64
  buckyball.fence
  func.call @check_result(%output) : (memref<1x16xi8>) -> ()
  buckyball.bank_release %loaded : i64
  buckyball.bank_release %stored : i64
  memref.dealloc %output : memref<1x16xi8>
  return %zero_i8 : i8
}
