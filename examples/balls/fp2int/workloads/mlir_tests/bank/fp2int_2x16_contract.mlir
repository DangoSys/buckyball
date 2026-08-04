// CHECK: buckyball.fp2int
// CHECK-NOT: buckyball.bank_fp2int

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 2 : i64
  %scale = arith.constant 1073741824 : i64
  %in = buckyball.bank_alloc {col = 4 : i64}
  %out = buckyball.bank_alloc
  %r = buckyball.bank_fp2int %in %out %iter %scale
      : i64 i64 i64 i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %r : i64
  return %z : i8
}
