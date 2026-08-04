// CHECK: buckyball.intr.custom
// CHECK-SAME: funct7 = 52
// CHECK-NOT: buckyball.int2fp

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 4 : i64
  %scale = arith.constant 1065353216 : i64
  %in = buckyball.bank_alloc
  %out = buckyball.bank_alloc
  buckyball.int2fp %in, %out, %iter, %scale : i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %out : i64
  return %z : i8
}
