// CHECK: buckyball.bank_mvin
// CHECK: buckyball.bank_im2col
// CHECK: buckyball.bank_mvout
// CHECK: buckyball.bank_mul_warp16

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_f32 = arith.constant 0.0 : f32
  %one_f32 = arith.constant 1.0 : f32

  %input = memref.alloc() : memref<1x8x8x16xf32>
  %filter = memref.alloc() : memref<3x3x16x32xf32>
  %output = memref.alloc() : memref<1x6x6x32xf32>

  linalg.fill ins(%one_f32 : f32) outs(%input : memref<1x8x8x16xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%filter : memref<3x3x16x32xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%output : memref<1x6x6x32xf32>)

  tile.tile_conv2d %input %filter %output
    : memref<1x8x8x16xf32> memref<3x3x16x32xf32> memref<1x6x6x32xf32>

  memref.dealloc %input : memref<1x8x8x16xf32>
  memref.dealloc %filter : memref<3x3x16x32xf32>
  memref.dealloc %output : memref<1x6x6x32xf32>

  return %zero_i8 : i8
}
