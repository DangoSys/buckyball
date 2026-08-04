// Int2FpBall convert op contract:
// - bank_int2fp lowers to int2fp
//
// RUN stages are driven by CMake FileCheck targets.
//
// CHECK-PHYSICAL: buckyball.int2fp
// CHECK-INTRIN: buckyball.intr.custom
// CHECK-INTRIN: funct7 = 52

func.func @int_convert_ops(%in0: i64, %out0: i64, %iter: i64, %scale: i64) {
  %r0 = buckyball.bank_int2fp %in0 %out0 %iter %scale
      : i64 i64 i64 i64
  return
}
