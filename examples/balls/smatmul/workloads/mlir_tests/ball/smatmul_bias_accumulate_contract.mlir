// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 17 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 65 : i32
// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 65 : i32
// CHECK-NOT: buckyball.smatmul
// CHECK-NOT: buckyball.bank_smatmul

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %cfg = arith.constant 268501008 : i64
  %bias = arith.constant 0 : i64
  %a0 = arith.constant 1 : i64
  %b0 = arith.constant 2 : i64
  %a1 = arith.constant 3 : i64
  %b1 = arith.constant 4 : i64
  %c = arith.constant 5 : i64
  buckyball.smatmul_bias %bias : i64
  buckyball.smatmul %a0, %b0, %c, %cfg {last = false} : i64
  buckyball.smatmul %a1, %b1, %c, %cfg {first = false} : i64
  return %z : i8
}
