func.func @i32_is_not_i8(%codes: tensor<1x16xi32>, %step: tensor<1x16xf32>) -> tensor<1x16xf32> {
  // expected-error @+1 {{operand #0 must be ranked tensor of 8-bit signless integer values}}
  %value = "frontend.int8_to_fp32"(%codes, %step) : (tensor<1x16xi32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  return %value : tensor<1x16xf32>
}

// -----

func.func @i8_is_not_i32(%codes: tensor<1x16xi8>, %step: tensor<1x16xf32>) -> tensor<1x16xf32> {
  // expected-error @+1 {{operand #0 must be ranked tensor of 32-bit signless integer values}}
  %value = "frontend.int32_to_fp32"(%codes, %step) : (tensor<1x16xi8>, tensor<1x16xf32>) -> tensor<1x16xf32>
  return %value : tensor<1x16xf32>
}

// -----

// VALID-LABEL: func.func @valid_formats
// VALID: arith.sitofp {{.*}} : i8 to f32
// VALID: arith.mulf
// VALID: arith.sitofp {{.*}} : i32 to f32
// VALID: arith.mulf
// VALID: return
func.func @valid_formats(%i8_codes: tensor<1x16xi8>, %i32_codes: tensor<1x16xi32>, %step: tensor<1x16xf32>) -> (tensor<1x16xf32>, tensor<1x16xf32>) {
  %first = "frontend.int8_to_fp32"(%i8_codes, %step) : (tensor<1x16xi8>, tensor<1x16xf32>) -> tensor<1x16xf32>
  %second = "frontend.int32_to_fp32"(%i32_codes, %step) : (tensor<1x16xi32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  return %first, %second : tensor<1x16xf32>, tensor<1x16xf32>
}

// -----

// A cycle trace calls the runtime around the traced operation; a tensor trace
// passes the flattened value at its end.
// VALID-DAG: func.func private @buddyTraceCycleEndPath(i64, i64, i64, i64, i64, i64) attributes {llvm.emit_c_interface}
// VALID-DAG: func.func private @buddyTraceCycleStartPath(i64, i64, i64, i64, i64, i64) attributes {llvm.emit_c_interface}
// VALID-DAG: func.func private @buddyTraceTensorF32Path(i64, i64, i64, i64, i64, i64, memref<?xf32>) attributes {llvm.emit_c_interface}
// VALID-LABEL: func.func @traced
// VALID: call @buddyTraceCycleStartPath
// VALID: linalg.generic
// VALID: call @buddyTraceCycleEndPath
// VALID: %[[BUFFER:.*]] = bufferization.to_buffer %{{.*}} read_only : tensor<2x3xf32> to memref<2x3xf32>
// VALID: %[[FLAT:.*]] = memref.collapse_shape %[[BUFFER]] {{\[\[}}0, 1]] : memref<2x3xf32> into memref<6xf32>
// VALID: %[[CAST:.*]] = memref.cast %[[FLAT]] : memref<6xf32> to memref<?xf32>
// VALID: call @buddyTraceTensorF32Path(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[CAST]])
// VALID: return
func.func @traced(%codes: tensor<2x3xi8>, %step: tensor<1x1xf32>) -> tensor<2x3xf32> {
  frontend.trace_start {cycle = true, id = 3 : i64, id_path = [3, 0], tag = "dequantize", tensor = true}
  %value = "frontend.int8_to_fp32"(%codes, %step) : (tensor<2x3xi8>, tensor<1x1xf32>) -> tensor<2x3xf32>
  %traced = frontend.trace_end %value {cycle = true, id = 3 : i64, id_path = [3, 0], tag = "dequantize", tensor = true} : tensor<2x3xf32>
  return %traced : tensor<2x3xf32>
}

// -----

func.func @dynamic_tensor_trace(%value: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{traces a tensor of no static shape}}
  %traced = frontend.trace_end %value {cycle = false, id = 4 : i64, id_path = [4], tensor = true} : tensor<?xf32>
  return %traced : tensor<?xf32>
}
