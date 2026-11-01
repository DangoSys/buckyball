// The mm_accumulate product of ctests/vectors.h (5x96 by 12x96 packed MXINT8
// onto an old 5x12 C) through the bank operation; the caller compares C with
// mm_accumulate_expected.
func.func @accumulate(%a: memref<35x16xi8>, %b: memref<84x16xi8>,
                      %c: memref<15x4xi32>) attributes {llvm.emit_c_interface} {
  %a_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = bbacc.bank_mvin %a, %a_bank <{depth = 35 : i64, stride = 1 : i64}> : memref<35x16xi8>
  %lb = bbacc.bank_mvin %b, %b_bank <{depth = 84 : i64, stride = 1 : i64}> : memref<84x16xi8>
  %lc = bbacc.bank_mvin %c, %c_bank <{depth = 15 : i64, stride = 1 : i64}> : memref<15x4xi32>
  %sum = mxint8matmul_ball.bank_matmul_mxint8 %la, %lb -> %lc <{m = 5 : i64, n = 12 : i64, k = 96 : i64, accumulate = true}>
  %stored = bbacc.bank_mvout %c, %sum <{depth = 15 : i64, stride = 1 : i64}> : memref<15x4xi32>
  bbacc.bank_release %la
  bbacc.bank_release %lb
  bbacc.bank_release %stored
  return
}
