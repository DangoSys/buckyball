// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 =
// CHECK-NOT: buckyball.smatmul

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %cfg = arith.constant 268501008 : i64
  %a = buckyball.bank_alloc
  %b = buckyball.bank_alloc
  %c = buckyball.bank_alloc {col = 2 : i64}
  buckyball.smatmul %a, %b, %c, %cfg : i64
  buckyball.bank_release %a : i64
  buckyball.bank_release %b : i64
  buckyball.bank_release %c : i64
  return %z : i8
}
