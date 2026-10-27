// CHECK-LABEL: func.func @define_quantized_rows
// CHECK: arith.constant 4298115073 : i64
// CHECK: return
func.func @define_quantized_rows() {
  toint8_ball.int32_to_int8 <{values = 1 : i64, scales = 2 : i64, output = 3 : i64, rows = 1 : i64}>
  return
}
