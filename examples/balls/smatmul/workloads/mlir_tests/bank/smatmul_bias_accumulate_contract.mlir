// CHECK: buckyball.smatmul_bias
// CHECK: buckyball.smatmul
// CHECK-SAME: last = false
// CHECK: buckyball.smatmul
// CHECK-SAME: first = false
// CHECK-NOT: buckyball.bank_smatmul

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %cfg = arith.constant 268501008 : i64
  %bias = buckyball.bank_alloc
  %a0 = buckyball.bank_alloc
  %b0 = buckyball.bank_alloc
  %a1 = buckyball.bank_alloc
  %b1 = buckyball.bank_alloc
  %c = buckyball.bank_alloc
  %biasState = buckyball.bank_smatmul_bias %bias : i64
  %partial = buckyball.bank_smatmul %a0 %b0 %c %cfg {last = false}
      : i64 i64 i64 i64
  %result = buckyball.bank_smatmul %a1 %b1 %partial %cfg {first = false}
      : i64 i64 i64 i64
  buckyball.bank_release %biasState : i64
  buckyball.bank_release %a0 : i64
  buckyball.bank_release %b0 : i64
  buckyball.bank_release %a1 : i64
  buckyball.bank_release %b1 : i64
  buckyball.bank_release %result : i64
  return %z : i8
}
