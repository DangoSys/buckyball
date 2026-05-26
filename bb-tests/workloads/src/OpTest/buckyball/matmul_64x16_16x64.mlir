// Buckyball Dialect matmul test: 64x16 @ 16x64
// Matrix: 64x16 (fp32) @ 16x64 (fp32) -> 64x64 (fp32)
// Tests buckyball.matmul lowering: fp32 -> quant -> mul_warp16 -> dequant
// Verification done in C wrapper to avoid fp constant pool (.LCPI) which
// triggers R_RISCV_HI20 relocation truncation under newlib.

func.func private @check_result(memref<64x64xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32

  %a = memref.alloc() : memref<64x16xf32>
  %b = memref.alloc() : memref<16x64xf32>
  %c = memref.alloc() : memref<64x64xf32>

  linalg.fill ins(%one_f32 : f32) outs(%a : memref<64x16xf32>)
  linalg.fill ins(%one_f32 : f32) outs(%b : memref<16x64xf32>)
  linalg.fill ins(%zero_f32 : f32) outs(%c : memref<64x64xf32>)

  buckyball.matmul %a %b %c
    : memref<64x16xf32> memref<16x64xf32> memref<64x64xf32>

  func.call @check_result(%c) : (memref<64x64xf32>) -> ()

  memref.dealloc %a : memref<64x16xf32>
  memref.dealloc %b : memref<16x64xf32>
  memref.dealloc %c : memref<64x64xf32>

  return %zero_i8 : i8
}
