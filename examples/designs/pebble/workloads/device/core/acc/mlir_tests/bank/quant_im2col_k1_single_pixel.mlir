// Design integration: f32-to-i8 quantization -> im2col k=1 -> mvout.

func.func private @check_result(memref<8x16xi8>) -> ()

memref.global "private" constant @fp_in : memref<4x4xf32> =
    dense<[[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]>

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %scale = arith.constant 1.0 : f32

  %input = memref.get_global @fp_in : memref<4x4xf32>
  %output = memref.alloc() alignment = 64 : memref<8x16xi8>

  %bin = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %bq = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %bout = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %loaded = buckyball.bank_mvin %input, %bin <{depth = 4 : i64, stride = 1 : i64}> : memref<4x4xf32>
  %q = toint8_ball.bank_fp32_to_int8 %loaded -> %bq multiplier(%scale) <{rows = 4 : i64}>
  %im = im2col_ball.bank_im2col_i8 %q -> %bout <{input_side = 1 : i64, kernel = 1 : i64, stride = 1 : i64, padding = 0 : i64, tile_rows = 8 : i64}>
  %stored = buckyball.bank_mvout %output, %im <{depth = 8 : i64, stride = 1 : i64}> : memref<8x16xi8>
  buckyball.bank_release %loaded
  buckyball.bank_release %q
  buckyball.bank_release %stored

  func.call @check_result(%output) : (memref<8x16xi8>) -> ()
  memref.dealloc %output : memref<8x16xi8>
  return %zero_i8 : i8
}
