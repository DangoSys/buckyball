// RUN: buddy-opt %s --target=toy -assign-physical-banks -split-input-file -verify-diagnostics
module {
  func.func private @wrong_type(i64, i64, i32)
  func.func @mismatched() {
    %read = buckyball.bank_alloc <col = 2>
    %write = buckyball.bank_alloc <col = 3>
    %one = arith.constant 1 : i64
    // expected-error @+1 {{bank kernel callee type does not match its operands}}
    %r, %w = buckyball.bank_kernel @wrong_type %read %write (%one) : i64
    buckyball.bank_release %r : i64
    buckyball.bank_release %w : i64
    return
  }
}
// -----
func.func @consumed_source() {
  %source = buckyball.bank_alloc
  %write = buckyball.bank_alloc <col = 3>
  %target = arith.constant 7 : i64
  %one = arith.constant 1 : i64
  // expected-error @+1 {{source bank handle is used after transfer or outside its block}}
  %aggregate = buckyball.bank_transfer %source %target : i64 i64
  %r, %w = buckyball.bank_kernel @mock %source %write (%one) : i64
  buckyball.bank_release %aggregate : i64
  buckyball.bank_release %w : i64
  return
}
// -----
func.func @unknown_read() {
  %read = arith.constant 7 : i64
  %write = buckyball.bank_alloc <col = 3>
  %one = arith.constant 1 : i64
  // expected-error @+1 {{bank kernel requires distinct live read/write handles}}
  %r, %w = buckyball.bank_kernel @mock %read %write (%one) : i64
  buckyball.bank_release %w : i64
  return
}
// -----
func.func @aliased_read_write() {
  %read = buckyball.bank_alloc <col = 2>
  %one = arith.constant 1 : i64
  // expected-error @+1 {{bank kernel requires distinct live read/write handles}}
  %r, %w = buckyball.bank_kernel @mock %read %read (%one) : i64
  buckyball.bank_release %w : i64
  return
}
