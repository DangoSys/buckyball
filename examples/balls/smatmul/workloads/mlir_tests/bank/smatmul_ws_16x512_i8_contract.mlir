// CHECK: buckyball.smatmul
// CHECK-NOT: buckyball.bank_smatmul

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %cfg = arith.constant 270532624 : i64
  %a = buckyball.bank_alloc
  %b = buckyball.bank_alloc
  %c = buckyball.bank_alloc {col = 2 : i64}
  %r = buckyball.bank_smatmul %a %b %c %cfg {ws = true}
      : i64 i64 i64 i64
  buckyball.bank_release %a : i64
  buckyball.bank_release %b : i64
  buckyball.bank_release %r : i64
  return %z : i8
}
