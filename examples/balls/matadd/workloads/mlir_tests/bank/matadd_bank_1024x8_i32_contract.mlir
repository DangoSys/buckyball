// CHECK: buckyball.matadd
// CHECK-NOT: buckyball.bank_matadd

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %lines = arith.constant 1024 : i64
  %a = buckyball.bank_alloc {col = 2 : i64}
  %b = buckyball.bank_alloc {col = 2 : i64}
  %c = buckyball.bank_alloc {col = 2 : i64}
  %result = buckyball.bank_matadd %a %b %c %lines : i64 i64 i64 i64
  buckyball.bank_release %a : i64
  buckyball.bank_release %b : i64
  buckyball.bank_release %result : i64
  return %zero : i8
}
