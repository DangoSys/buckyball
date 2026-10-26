// RUN: buddy-opt %s -split-input-file -assign-physical-banks -lower-buckyball -verify-diagnostics
func.func @same_bank(%input: memref<1x4xi32>) {
  %a = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = buckyball.bank_mvin %input, %a <{depth = 1 : i64, stride = 1 : i64}> : memref<1x4xi32>
  // expected-error @+2 {{add_f32 requires distinct banks, rows within a bank and a group in [0, 31]}}
  // expected-error @+1 {{has no lowering on this core}}
  %out = buckyball.bank_custom "add_f32" (%la, %b) -> %b {rows = 1 : i64, group = 0 : i64, first = true}
  buckyball.bank_release %la
  buckyball.bank_release %out
  return
}
// -----
func.func @zero_rows(%input: memref<1x4xi32>) {
  %a = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = buckyball.bank_mvin %input, %a <{depth = 1 : i64, stride = 1 : i64}> : memref<1x4xi32>
  // expected-error @+2 {{add_f32 requires distinct banks, rows within a bank and a group in [0, 31]}}
  // expected-error @+1 {{has no lowering on this core}}
  %out = buckyball.bank_custom "add_f32" (%la, %b) -> %c {rows = 0 : i64, group = 0 : i64, first = true}
  buckyball.bank_release %la
  buckyball.bank_release %b
  buckyball.bank_release %out
  return
}
// -----
func.func @negative_group(%input: memref<1x4xi32>) {
  %a = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %b = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %c = buckyball.bank_alloc <{row = 1 : i64, col = 1 : i64}>
  %la = buckyball.bank_mvin %input, %a <{depth = 1 : i64, stride = 1 : i64}> : memref<1x4xi32>
  // expected-error @+2 {{add_f32 requires distinct banks, rows within a bank and a group in [0, 31]}}
  // expected-error @+1 {{has no lowering on this core}}
  %out = buckyball.bank_custom "add_f32" (%la, %b) -> %c {rows = 1 : i64, group = -1 : i64, first = false}
  buckyball.bank_release %la
  buckyball.bank_release %b
  buckyball.bank_release %out
  return
}
