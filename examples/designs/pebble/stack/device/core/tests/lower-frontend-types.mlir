func.func @i32_is_not_i8(%codes: tensor<1x16xi32>, %step: tensor<1x16xf32>) -> tensor<1x16xf32> {
  // expected-error @+1 {{requires an INT8 input}}
  %value = "frontend.int8_to_fp32"(%codes, %step) : (tensor<1x16xi32>, tensor<1x16xf32>) -> tensor<1x16xf32>
  return %value : tensor<1x16xf32>
}

// -----

func.func @i8_is_not_i32(%codes: tensor<1x16xi8>, %step: tensor<1x16xf32>) -> tensor<1x16xf32> {
  // expected-error @+1 {{requires an INT32 input}}
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
