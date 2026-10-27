// One MXFP8 step on columns 32..63 of a one-row A of 64 columns. C comes in
// holding a guard pattern and goes back with the row of 16 sums.
func.func @window(%a: memref<5x16xi8>, %b: memref<33x16xi8>,
                  %c: memref<5x16xi8>) attributes {llvm.emit_c_interface} {
  %a_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c_bank = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = buckyball.bank_mvin %a, %a_bank <{depth = 5 : i64, stride = 1 : i64}> : memref<5x16xi8>
  %lb = buckyball.bank_mvin %b, %b_bank <{depth = 33 : i64, stride = 1 : i64}> : memref<33x16xi8>
  %lc = buckyball.bank_mvin %c, %c_bank <{depth = 5 : i64, stride = 1 : i64}> : memref<5x16xi8>
  %sum = mxmm_ball.bank_matmul_mxfp8_window %la, %lb -> %lc <{rows = 1 : i64, cols = 16 : i64, k = 32 : i64, full_k = 64 : i64, start_k = 32 : i64, base = 0 : i64, first, last}>
  %stored = buckyball.bank_mvout %c, %sum <{depth = 5 : i64, stride = 1 : i64}> : memref<5x16xi8>
  buckyball.bank_release %la
  buckyball.bank_release %lb
  buckyball.bank_release %stored
  return
}
