// Two steps of one MXFP8 chain over the same A and B: the first only
// accumulates, the last accumulates and writes the 16x16 sum to C from row 1.
// C comes in holding a guard pattern and goes back with the sum.
func.func @chain(%a: memref<33x16xi8>, %b: memref<33x16xi8>,
                 %c: memref<65x16xi8>) attributes {llvm.emit_c_interface} {
  %a_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = buckyball.bank_mvin %a, %a_bank <{depth = 33 : i64, stride = 1 : i64}> : memref<33x16xi8>
  %lb = buckyball.bank_mvin %b, %b_bank <{depth = 33 : i64, stride = 1 : i64}> : memref<33x16xi8>
  %lc = buckyball.bank_mvin %c, %c_bank <{depth = 65 : i64, stride = 1 : i64}> : memref<65x16xi8>
  %partial = mxmm_ball.bank_matmul_mxfp8 %la, %lb -> %lc <{rows = 16 : i64, cols = 16 : i64, k = 32 : i64, base = 1 : i64, first}>
  %sum = mxmm_ball.bank_matmul_mxfp8 %la, %lb -> %partial <{rows = 16 : i64, cols = 16 : i64, k = 32 : i64, base = 1 : i64, last}>
  %stored = buckyball.bank_mvout %c, %sum <{depth = 65 : i64, stride = 1 : i64}> : memref<65x16xi8>
  buckyball.bank_release %la
  buckyball.bank_release %lb
  buckyball.bank_release %stored
  return
}
