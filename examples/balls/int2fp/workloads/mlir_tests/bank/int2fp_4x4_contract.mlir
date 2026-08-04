// CHECK: buckyball.int2fp
// CHECK-NOT: buckyball.bank_int2fp

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 4 : i64
  %scale = arith.constant 1065353216 : i64
  %in = buckyball.bank_alloc
  %out = buckyball.bank_alloc
  %r = buckyball.bank_int2fp %in %out %iter %scale
      : i64 i64 i64 i64
  buckyball.bank_release %in : i64
  buckyball.bank_release %r : i64
  return %z : i8
}
