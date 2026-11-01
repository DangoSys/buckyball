// One unfused FP32 step that both starts and ends its chain. C comes in holding
// a guard pattern and goes back with the 16x16 sum.
func.func @single(%a: memref<16x4xi32>, %b: memref<16x4xi32>,
                  %c: memref<65x4xi32>) attributes {llvm.emit_c_interface} {
  %a_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c_bank = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = bbacc.bank_mvin %a, %a_bank <{depth = 16 : i64, stride = 1 : i64}> : memref<16x4xi32>
  %lb = bbacc.bank_mvin %b, %b_bank <{depth = 16 : i64, stride = 1 : i64}> : memref<16x4xi32>
  %lc = bbacc.bank_mvin %c, %c_bank <{depth = 65 : i64, stride = 1 : i64}> : memref<65x4xi32>
  %sum = mxmm_ball.bank_matmul_f32 %la, %lb -> %lc <{rows = 16 : i64, cols = 16 : i64, k = 4 : i64, base = 0 : i64, first, last}>
  %stored = bbacc.bank_mvout %c, %sum <{depth = 65 : i64, stride = 1 : i64}> : memref<65x4xi32>
  bbacc.bank_release %la
  bbacc.bank_release %lb
  bbacc.bank_release %stored
  return
}
