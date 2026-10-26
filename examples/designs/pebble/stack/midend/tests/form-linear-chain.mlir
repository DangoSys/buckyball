// FORM-LABEL: func.func @cycle_chain
// FORM: %[[FIRST:.*]] = custom.op "linear_i8"{{.*}}custom.chain = [[CHAIN:[0-9]+]] : i64, custom.relu = true
// FORM: %[[TRACE:.*]] = buddy_trace.end %[[FIRST]] {{.*}}custom.chain = [[CHAIN]] : i64{{.*}}tensor = false
// FORM: %[[RELU:.*]] = custom.op "relu"(%[[TRACE]]) {{.*}}custom.chain = [[CHAIN]] : i64
// FORM: custom.op "linear_i8"(%[[RELU]], {{.*}}custom.chain = [[CHAIN]] : i64
// FORM: return
// BANK-LABEL: func.func @cycle_chain
// BANK-NOT: {{fp32_to_int8|int32_to_int8|int32_to_fp32|bank_mvout}}
// BANK: buckyball.bank_custom "fp32_to_int8"
// BANK-NOT: {{fp32_to_int8|int32_to_int8|int32_to_fp32|bank_mvout}}
// BANK: buckyball.bank_custom "int32_to_int8"{{.*}}relu = true{{.*}}wr_base = 0 : i64
// BANK-NOT: {{fp32_to_int8|int32_to_int8|int32_to_fp32|bank_mvout}}
// BANK: %[[RESIDENT:.*]] = buckyball.bank_custom "int32_to_int8"{{.*}}relu = true{{.*}}wr_base = 1 : i64
// BANK-NOT: {{fp32_to_int8|int32_to_int8|int32_to_fp32|bank_mvout}}
// BANK: buddy_trace.end %[[RESIDENT]] {{.*}}tensor = false} : i64
// BANK-NOT: {{fp32_to_int8|int32_to_int8|int32_to_fp32|bank_mvout}}
// BANK: buckyball.bank_custom "int32_to_fp32"
// BANK-NOT: {{fp32_to_int8|int32_to_int8|int32_to_fp32|bank_mvout}}
// BANK: buckyball.bank_mvout
// BANK-NOT: {{fp32_to_int8|int32_to_int8|int32_to_fp32|bank_mvout}}
// BANK: return
func.func @cycle_chain(%x: tensor<1x32xf32>, %w1: tensor<32x32xi8>, %b1: tensor<1x32xi32>, %s1: tensor<1x32xf32>, %w2: tensor<32x16xi8>, %b2: tensor<1x16xi32>, %s2: tensor<1x16xf32>, %multiplier: tensor<1x1xf32>, %step: tensor<1x1xf32>) -> tensor<1x16xf32> {
  %zero = arith.constant dense<0.0> : tensor<1x32xf32>
  %empty = tensor.empty() : tensor<1x32xf32>
  %first = custom.op "linear_i8"(%x, %w1, %b1, %multiplier, %step, %s1) : (tensor<1x32xf32>, tensor<32x32xi8>, tensor<1x32xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  %trace = buddy_trace.end %first {cycle = true, id = 0 : i64, id_path = [0], tag = "first", tensor = false} : tensor<1x32xf32>
  %relu = linalg.generic {
    indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%trace, %zero: tensor<1x32xf32>, tensor<1x32xf32>) outs(%empty: tensor<1x32xf32>) {
  ^bb0(%value: f32, %zero_value: f32, %unused: f32):
    %positive = arith.maximumf %value, %zero_value : f32
    linalg.yield %positive : f32
  } -> tensor<1x32xf32>
  %second = custom.op "linear_i8"(%relu, %w2, %b2, %multiplier, %step, %s2) : (tensor<1x32xf32>, tensor<32x16xi8>, tensor<1x16xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  return %second : tensor<1x16xf32>
}

// FORM-LABEL: func.func @tensor_trace
// FORM: custom.op "linear_i8"
// FORM: buddy_trace.end {{.*}}tensor = true} : tensor<1x32xf32>
// FORM: linalg.generic
// FORM: arith.maximumf
// FORM: custom.op "linear_i8"
// FORM: return
func.func @tensor_trace(%x: tensor<1x32xf32>, %w1: tensor<32x32xi8>, %b1: tensor<1x32xi32>, %s1: tensor<1x32xf32>, %w2: tensor<32x16xi8>, %b2: tensor<1x16xi32>, %s2: tensor<1x16xf32>, %multiplier: tensor<1x1xf32>, %step: tensor<1x1xf32>) -> tensor<1x16xf32> {
  %zero = arith.constant dense<0.0> : tensor<1x32xf32>
  %empty = tensor.empty() : tensor<1x32xf32>
  %first = custom.op "linear_i8"(%x, %w1, %b1, %multiplier, %step, %s1) : (tensor<1x32xf32>, tensor<32x32xi8>, tensor<1x32xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  %trace = buddy_trace.end %first {cycle = true, id = 1 : i64, id_path = [1], tag = "first", tensor = true} : tensor<1x32xf32>
  %relu = linalg.generic {
    indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%trace, %zero: tensor<1x32xf32>, tensor<1x32xf32>) outs(%empty: tensor<1x32xf32>) {
  ^bb0(%value: f32, %zero_value: f32, %unused: f32):
    %positive = arith.maximumf %value, %zero_value : f32
    linalg.yield %positive : f32
  } -> tensor<1x32xf32>
  %second = custom.op "linear_i8"(%relu, %w2, %b2, %multiplier, %step, %s2) : (tensor<1x32xf32>, tensor<32x16xi8>, tensor<1x16xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  return %second : tensor<1x16xf32>
}

// FORM-LABEL: func.func @multiple_consumers
// FORM: %[[OBSERVED:.*]] = custom.op "linear_i8"
// FORM: linalg.generic
// FORM: arith.maximumf
// FORM: custom.op "linear_i8"
// FORM: return {{.*}}, %[[OBSERVED]]
func.func @multiple_consumers(%x: tensor<1x32xf32>, %w1: tensor<32x32xi8>, %b1: tensor<1x32xi32>, %s1: tensor<1x32xf32>, %w2: tensor<32x16xi8>, %b2: tensor<1x16xi32>, %s2: tensor<1x16xf32>, %multiplier: tensor<1x1xf32>, %step: tensor<1x1xf32>) -> (tensor<1x16xf32>, tensor<1x32xf32>) {
  %zero = arith.constant dense<0.0> : tensor<1x32xf32>
  %empty = tensor.empty() : tensor<1x32xf32>
  %first = custom.op "linear_i8"(%x, %w1, %b1, %multiplier, %step, %s1) : (tensor<1x32xf32>, tensor<32x32xi8>, tensor<1x32xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  %relu = linalg.generic {
    indexing_maps = [affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>, affine_map<(m,n)->(m,n)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%first, %zero: tensor<1x32xf32>, tensor<1x32xf32>) outs(%empty: tensor<1x32xf32>) {
  ^bb0(%value: f32, %zero_value: f32, %unused: f32):
    %positive = arith.maximumf %value, %zero_value : f32
    linalg.yield %positive : f32
  } -> tensor<1x32xf32>
  %second = custom.op "linear_i8"(%relu, %w2, %b2, %multiplier, %step, %s2) : (tensor<1x32xf32>, tensor<32x16xi8>, tensor<1x16xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  return %second, %first : tensor<1x16xf32>, tensor<1x32xf32>
}
