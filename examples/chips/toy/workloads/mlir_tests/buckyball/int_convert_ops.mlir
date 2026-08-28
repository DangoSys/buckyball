// Int2FpBall convert op contract:
// - bank_int2fp_tensor lowers to int2fp_tensor
//
// RUN stages are driven by CMake FileCheck targets.
//
// CHECK-PHYSICAL: buckyball.int2fp_tensor
// CHECK-INTRIN: buckyball.intr.custom
// CHECK-INTRIN: funct7 = 52

func.func @int_convert_ops(%in0: i64, %out0: i64, %iter: i64, %da: i64, %dw: i64) {
  %r0 = buckyball.bank_int2fp_tensor %in0 %out0 %iter %da %dw
      : i64 i64 i64 i64 i64
  return
}
