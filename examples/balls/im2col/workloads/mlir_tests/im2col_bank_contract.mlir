// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 48
// CHECK-NOT: buckyball.bank_im2col

// Contract shape within default.toml: iter/ksize <= 7, padding <= 3.
func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %iter = arith.constant 7 : i64
  %ksize = arith.constant 7 : i64
  %stride = arith.constant 1 : i64
  %padding = arith.constant 0 : i64

  %in = buckyball.bank_alloc
  %out = buckyball.bank_alloc
  %next = buckyball.bank_im2col %in %out %iter %ksize %stride %padding
    : i64 i64 i64 i64 i64 i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %next : i64

  return %zero_i8 : i8
}
