// Two steps of one MXFP8 chain over the same A and B: the first only
// accumulates, the last accumulates and writes the 16x16 sum to C from row 1.
// C comes in holding a guard pattern and goes back with the sum.
func.func @chain(%a: memref<33x16xi8>, %b: memref<33x16xi8>,
                 %c: memref<65x16xi8>) attributes {llvm.emit_c_interface} {
  %a_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = bbacc.bank_mvin %a, %a_bank <{depth = 33 : i64, stride = 1 : i64}> : memref<33x16xi8>
  %lb = bbacc.bank_mvin %b, %b_bank <{depth = 33 : i64, stride = 1 : i64}> : memref<33x16xi8>
  %lc = bbacc.bank_mvin %c, %c_bank <{depth = 65 : i64, stride = 1 : i64}> : memref<65x16xi8>
  %partial = mxmm_ball.bank_matmul_mxfp8 %la, %lb -> %lc <{rows = 16 : i64, cols = 16 : i64, k = 32 : i64, base = 1 : i64, first}>
  %sum = mxmm_ball.bank_matmul_mxfp8 %la, %lb -> %partial <{rows = 16 : i64, cols = 16 : i64, k = 32 : i64, base = 1 : i64, last}>
  %stored = bbacc.bank_mvout %c, %sum <{depth = 65 : i64, stride = 1 : i64}> : memref<65x16xi8>
  bbacc.bank_release %la
  bbacc.bank_release %lb
  bbacc.bank_release %stored
  return
}
