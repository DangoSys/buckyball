// CHECK-LABEL: func.func @named
// CHECK: buckyball.bank_mvin
// CHECK-NOT: "fp32_to_int8"
// CHECK: buckyball.bank_custom "matmul_i8"
// CHECK: buckyball.bank_mvout
// CHECK: buckyball.bank_custom "matmul_i8"
// CHECK: buckyball.bank_mvout
// CHECK: return
func.func @named(%a: tensor<1x32xi8>, %b: tensor<32x32xi8>, %initial: tensor<1x32xi32>) -> tensor<1x32xi32> {
  %product = custom.op "matmul_i8"(%a, %b, %initial) : (tensor<1x32xi8>, tensor<32x32xi8>, tensor<1x32xi32>) -> tensor<1x32xi32>
  return %product : tensor<1x32xi32>
}

// CHECK-LABEL: func.func @linear
// CHECK: buckyball.bank_custom "fp32_to_int8"
// CHECK: buckyball.bank_custom "matmul_i8"
// CHECK: buckyball.bank_custom "int32_to_fp32"
// CHECK: buckyball.bank_mvout
// CHECK: buckyball.bank_custom "matmul_i8"
// CHECK: buckyball.bank_custom "int32_to_fp32"
// CHECK: buckyball.bank_mvout
// CHECK: return
func.func @linear(%a: tensor<1x32xf32>, %w: tensor<32x32xi8>, %bias: tensor<1x32xi32>, %multiplier: tensor<1x1xf32>, %input_step: tensor<1x1xf32>, %weight_step: tensor<1x32xf32>) -> tensor<1x32xf32> {
  %product = custom.op "linear_i8"(%a, %w, %bias, %multiplier, %input_step, %weight_step) : (tensor<1x32xf32>, tensor<32x32xi8>, tensor<1x32xi32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>
  return %product : tensor<1x32xf32>
}
