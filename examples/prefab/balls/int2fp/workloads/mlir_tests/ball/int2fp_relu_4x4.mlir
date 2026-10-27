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
  buckyball.mset <{bank = 1 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = true, row = 1 : i64, col = 1 : i64}>
  buckyball.mvin %input <{bank = 1 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xi32>
  buckyball.mvin %scales <{bank = 2 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xf32>
  int2fp_ball.int32_to_fp32 <{values = 1 : i64, scales = 2 : i64, output = 3 : i64, rows = 1 : i64, relu}>
  buckyball.mvout %output <{bank = 3 : i64, depth = 4 : i64, stride = 1 : i64}> : memref<4x4xf32>
  func.call @check_result(%output) : (memref<4x4xf32>) -> ()
  buckyball.mset <{bank = 1 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 2 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  buckyball.mset <{bank = 3 : i64, alloc = false, row = 0 : i64, col = 0 : i64}>
  memref.dealloc %output : memref<4x4xf32>
  return %zero_i8 : i8
}
