// RUN: buddy-opt %s -split-input-file -assign-physical-banks -lower-acc -verify-diagnostics
func.func @same_bank(%input: memref<1x4xi32>) {
  %a = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = bbacc.bank_mvin %input, %a <{depth = 1 : i64, stride = 1 : i64}> : memref<1x4xi32>
  // expected-error @+1 {{add_f32 requires distinct A, B and C banks and rows within a bank}}
  %out = f32add_ball.bank_add_f32 %la, %b -> %b <{rows = 1 : i64}>
  bbacc.bank_release %la
  bbacc.bank_release %out
  return
}
// -----
func.func @zero_rows(%input: memref<1x4xi32>) {
  %a = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c = bbacc.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = bbacc.bank_mvin %input, %a <{depth = 1 : i64, stride = 1 : i64}> : memref<1x4xi32>
  // expected-error @+1 {{add_f32 requires distinct A, B and C banks and rows within a bank}}
  %out = f32add_ball.bank_add_f32 %la, %b -> %c <{rows = 0 : i64}>
  bbacc.bank_release %la
  bbacc.bank_release %b
  bbacc.bank_release %out
  return
}
