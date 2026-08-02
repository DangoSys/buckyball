// CHECK: buckyball.im2col
// CHECK-NOT: buckyball.bank_im2col

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 6 : i64
  %ksize = arith.constant 3 : i64
  %stride = arith.constant 1 : i64
  %padding = arith.constant 0 : i64
  %in = buckyball.bank_alloc
  %out = buckyball.bank_alloc
  %r = buckyball.bank_im2col %in %out %iter %ksize %stride %padding
      : i64 i64 i64 i64 i64 i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %r : i64
  return %z : i8
}
