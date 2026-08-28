// CHECK: buckyball.int2fp_channel
// CHECK-NOT: buckyball.bank_int2fp_channel

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 1 : i64
  %da = arith.constant 0 : i64
  %dw = arith.constant 16 : i64
  %in = buckyball.bank_alloc {col = 4 : i64}
  %out = buckyball.bank_alloc {col = 4 : i64}
  %r = buckyball.bank_int2fp_channel %in %out %iter %da %dw
      : i64 i64 i64 i64 i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %r : i64
  return %z : i8
}
