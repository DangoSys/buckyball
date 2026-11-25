// The mm_accumulate product of the C tests (5x96 by 12x96 packed MXINT8
// onto an old 5x12 C) through the bank operation; the caller compares C with
// the precomputed exact block-ordered FP32 product.
func.func @accumulate(%a: memref<?x16xi8>, %b: memref<84x16xi8>,
                      %c: memref<15x4xi32>) attributes {llvm.emit_c_interface} {
  %c0 = arith.constant 0 : index
  %c7 = arith.constant 7 : index
  %rows = memref.dim %a, %c0 : memref<?x16xi8>
  %m = arith.divui %rows, %c7 : index
  %a_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = bbacc.bank_mvin %a, %a_bank <{depth = 35 : i64, stride = 1 : i64}> : memref<?x16xi8>
  %lb = bbacc.bank_mvin %b, %b_bank <{depth = 84 : i64, stride = 1 : i64}> : memref<84x16xi8>
  %lc = bbacc.bank_mvin %c, %c_bank <{depth = 15 : i64, stride = 1 : i64}> : memref<15x4xi32>
  %sum = mxint8matmul_ball.bank_matmul_mxint8 %la, %lb -> %lc rows %m <{n = 12 : i64, k = 96 : i64, accumulate = true}>
  %stored = bbacc.bank_mvout %c, %sum <{depth = 15 : i64, stride = 1 : i64}> : memref<15x4xi32>
  // The verification readback must wait for the ordinary DMA store.
  llvm.inline_asm has_side_effects "fence", "~{memory}" : () -> ()
  %readback = bbacc.bank_mvin %c, %lb <{depth = 15 : i64, stride = 1 : i64}> : memref<15x4xi32>
  bbacc.bank_release %la
  bbacc.bank_release %stored
  bbacc.bank_release %readback
  return
}
