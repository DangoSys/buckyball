// Int2FpBall convert op contract:
// - bank_int2fp lowers to int2fp (output_mode=0)
// - bank_int32_to_int8 lowers to int32_to_int8 (output_mode=1)
//
// RUN stages are driven by CMake FileCheck targets.
//
// CHECK-PHYSICAL: buckyball.int2fp
// CHECK-PHYSICAL: buckyball.int32_to_int8
// CHECK-INTRIN: buckyball.intr.custom
// CHECK-INTRIN: funct7 = 52
// CHECK-INTRIN: arith.constant 4294967296

func.func @int_convert_ops(%in0: i64, %out0: i64, %in1: i64, %out1: i64, %iter: i64, %scale: i64) {
  %r0 = buckyball.bank_int2fp %in0 %out0 %iter %scale
      : i64 i64 i64 i64
  %r1 = buckyball.bank_int32_to_int8 %in1 %out1 %iter %scale
      : i64 i64 i64 i64
  return
}
