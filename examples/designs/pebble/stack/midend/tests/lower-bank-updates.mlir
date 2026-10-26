// CHECK-LABEL: func.func @copy_rows
// CHECK: arith.constant -9223372035778936831 : i64
// CHECK: return
func.func @copy_rows() {
  buckyball.custom "maxpool_i8" <sources = [1], destination = 2, read_destination> {input_side = 1 : i64, output_side = 1 : i64, kernel = 1 : i64, stride = 1 : i64, padding = 0 : i64, op1_base = 0 : i64, wr_base = 1 : i64, output_stride = 1 : i64, start_row = 0 : i64, start_col = 0 : i64}
  return
}
// CHECK-LABEL: func.func @update_quantized_rows
// CHECK: arith.constant -9223372032556660735 : i64
// CHECK: return
func.func @update_quantized_rows() {
  buckyball.custom "int32_to_int8" <sources = [1, 2], destination = 3, read_destination> {rows = 1 : i64, wr_base = 1 : i64, relu = false}
  return
}
// CHECK-LABEL: func.func @define_quantized_rows
// CHECK: arith.constant 4298115073 : i64
// CHECK: return
func.func @define_quantized_rows() {
  buckyball.custom "int32_to_int8" <sources = [1, 2], destination = 3> {rows = 1 : i64, wr_base = 0 : i64, relu = false}
  return
}
