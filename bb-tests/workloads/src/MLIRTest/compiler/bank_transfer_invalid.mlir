// RUN: buddy-opt %s --target=toy -assign-physical-banks -split-input-file -verify-diagnostics

func.func @source_use_after_transfer() {
  %source = buckyball.bank_alloc
  %target = arith.constant 2 : i64
  // expected-error @+1 {{source bank handle is used after transfer or outside its block}}
  %aggregate = buckyball.bank_transfer %source %target : i64 i64
  buckyball.bank_release %source : i64
  buckyball.bank_release %aggregate : i64
  return
}
// -----
func.func @self() {
  %source = buckyball.bank_alloc
  // expected-error @+1 {{bank transfer source and target must differ}}
  %aggregate = buckyball.bank_transfer %source %source : i64 i64
  buckyball.bank_release %aggregate : i64
  return
}
// -----
func.func @unknown() {
  %source = arith.constant 7 : i64
  %target = arith.constant 2 : i64
  // expected-error @+1 {{transfer consumes an unknown virtual bank handle}}
  %aggregate = buckyball.bank_transfer %source %target : i64 i64
  buckyball.bank_release %aggregate : i64
  return
}
// -----
func.func @keep_other_owner() {
  %source = buckyball.bank_alloc
  %keep = buckyball.bank_alloc
  %target = buckyball.bank_alloc <col = 14>
  %aggregate = buckyball.bank_transfer %source %target : i64 i64
  buckyball.bank_release %aggregate : i64
  %all_free = buckyball.bank_alloc <col = 15>
  // expected-error @+1 {{unavailable bank resources (request=1x1, used=16/16, private ID max=15)}}
  %extra = buckyball.bank_alloc
  buckyball.bank_release %extra : i64
  buckyball.bank_release %all_free : i64
  buckyball.bank_release %keep : i64
  return
}
// -----
func.func @private_overflow() {
  %source = buckyball.bank_alloc
  %target = arith.constant 16 : i64
  // expected-error @+1 {{bank transfer target exceeds the private virtual ID range}}
  %aggregate = buckyball.bank_transfer %source %target : i64 i64
  buckyball.bank_release %aggregate : i64
  return
}
// -----
func.func @old_source_alias_after_id_reuse() {
  %source = buckyball.bank_alloc
  %target = arith.constant 2 : i64
  // expected-error @+1 {{source bank handle is used after transfer or outside its block}}
  %aggregate = buckyball.bank_transfer %source %target : i64 i64
  %new_source_id = buckyball.bank_alloc
  buckyball.bank_release %source : i64
  buckyball.bank_release %new_source_id : i64
  buckyball.bank_release %aggregate : i64
  return
}
