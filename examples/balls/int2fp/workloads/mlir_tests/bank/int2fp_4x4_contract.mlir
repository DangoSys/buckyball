// CHECK: buckyball.int2fp
// CHECK: buckyball.int32_to_int8
// CHECK-NOT: buckyball.bank_int2fp
// CHECK-NOT: buckyball.bank_int32_to_int8

func.func @main() -> i8 {
  %z = arith.constant 0 : i8
  %iter = arith.constant 4 : i64
  %scale = arith.constant 1065353216 : i64
  %in0 = buckyball.bank_alloc
  %out0 = buckyball.bank_alloc
  %in1 = buckyball.bank_alloc
  %out1 = buckyball.bank_alloc
  %r0 = buckyball.bank_int2fp %in0 %out0 %iter %scale
      : i64 i64 i64 i64
  %r1 = buckyball.bank_int32_to_int8 %in1 %out1 %iter %scale
      : i64 i64 i64 i64
  buckyball.bank_release %in0 : i64
  buckyball.bank_release %r0 : i64
  buckyball.bank_release %in1 : i64
  buckyball.bank_release %r1 : i64
  return %z : i8
}
