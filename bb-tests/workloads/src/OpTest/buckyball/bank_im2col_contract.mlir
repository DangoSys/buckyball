// CHECK: buckyball.intr.im2col
// CHECK-NOT: buckyball.bank_im2col

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %krow = arith.constant 3 : i64
  %kcol = arith.constant 3 : i64
  %inrow = arith.constant 8 : i64
  %incol = arith.constant 128 : i64
  %startrow = arith.constant 0 : i64
  %startcol = arith.constant 0 : i64

  %in = buckyball.bank_alloc
  %out = buckyball.bank_alloc
  %next = buckyball.bank_im2col %in %out %krow %kcol %inrow %incol %startrow %startcol
    : i64 i64 i64 i64 i64 i64 i64 i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %next : i64

  return %zero_i8 : i8
}
