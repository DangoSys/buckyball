// CHECK: buckyball.transpose
// CHECK-NOT: buckyball.bank_transpose

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 16 : i64
  %elem = arith.constant 8 : i64
  %mask = arith.constant -1 : i64
  %in = buckyball.bank_alloc
  %out = buckyball.bank_alloc
  %r = buckyball.bank_transpose %in %out %iter %elem %mask
      : i64 i64 i64 i64 i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %r : i64
  return %z : i8
}
