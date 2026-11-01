// Quantizes the quant_random rows of ctests/vectors.h (3 rows of 544 FP32)
// through the bank operation; the caller compares the packed rows with
// quant_random_expected.
func.func @quant(%input: memref<408x4xi32>, %output: memref<108x16xi8>)
    attributes {llvm.emit_c_interface} {
  %in_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %out_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %li = bbacc.bank_mvin %input, %in_bank <{depth = 408 : i64, stride = 1 : i64}> : memref<408x4xi32>
  %packed = mxint8quantize_ball.bank_quant_mxint8 %li -> %out_bank <{rows = 3 : i64, k = 544 : i64}>
  %stored = bbacc.bank_mvout %output, %packed <{depth = 108 : i64, stride = 1 : i64}> : memref<108x16xi8>
  bbacc.bank_release %li
  bbacc.bank_release %stored
  return
}
