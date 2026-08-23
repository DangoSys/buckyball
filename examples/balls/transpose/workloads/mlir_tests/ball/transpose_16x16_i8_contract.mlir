// CHECK: buckyball.intr.custom
// CHECK-NOT: buckyball.transpose

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 16 : i64
  %elem = arith.constant 8 : i64
  %in = buckyball.bank_alloc
  %out = buckyball.bank_alloc
  buckyball.transpose %in, %out, %iter, %elem : i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %out : i64
  return %z : i8
}
