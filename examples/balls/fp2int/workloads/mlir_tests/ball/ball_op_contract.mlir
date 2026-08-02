// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 51
// CHECK-NOT: buckyball.fp2int

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 4 : i64
  %scale = arith.constant 1065353216 : i64
  %in = buckyball.bank_alloc
  %out = buckyball.bank_alloc
  buckyball.fp2int %in, %out, %iter, %scale : i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %out : i64
  return %z : i8
}
