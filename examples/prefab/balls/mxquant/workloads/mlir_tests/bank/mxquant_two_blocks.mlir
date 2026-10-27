// Quantizes two blocks of 32 FP32. The output comes in holding a guard
// pattern and goes back with the codes and scales.
func.func @quant(%input: memref<16x4xi32>, %output: memref<5x16xi8>)
    attributes {llvm.emit_c_interface} {
  %in_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %out_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %li = buckyball.bank_mvin %input, %in_bank <{depth = 16 : i64, stride = 1 : i64}> : memref<16x4xi32>
  %lo = buckyball.bank_mvin %output, %out_bank <{depth = 5 : i64, stride = 1 : i64}> : memref<5x16xi8>
  %packed = mxquant_ball.bank_quant_mxfp8 %li -> %lo <{count = 64 : i64}>
  %stored = buckyball.bank_mvout %output, %packed <{depth = 5 : i64, stride = 1 : i64}> : memref<5x16xi8>
  buckyball.bank_release %li
  buckyball.bank_release %stored
  return
}
