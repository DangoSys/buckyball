func.func private @check_result(memref<1x16xi8>) -> ()

memref.global "private" constant @input : memref<4x4xf32> = dense<[
  [-3.000000e+02, -2.550000e+02, -5.000000e+00, -3.000000e+00],
  [-1.000000e+00, 0.000000e+00, 1.000000e+00, 3.000000e+00],
  [5.000000e+00, 2.540000e+02, 2.550000e+02, 3.000000e+02],
  [2.000000e+00, 6.000000e+00, 1.000000e+01, 1.400000e+01]
]>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %scale = arith.constant 5.000000e-01 : f32
  %input = memref.get_global @input : memref<4x4xf32>
  %output = memref.alloc() alignment = 64 : memref<1x16xi8>
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 1 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xf32>
  toint8_ball.fp32_to_int8 multiplier(%scale) <{input = 1 : i64, output = 2 : i64, rows = 4 : i64}>
  buckyball.mvout %output <{bank = 2 : i64, depth = 1 : i64, stride = 1 : i64}> : memref<1x16xi8>
  func.call @check_result(%output) : (memref<1x16xi8>) -> ()
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  memref.dealloc %output : memref<1x16xi8>
  return %zero_i8 : i8
}
