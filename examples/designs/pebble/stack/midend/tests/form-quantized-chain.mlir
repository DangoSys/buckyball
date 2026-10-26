// FORM-LABEL: func.func @squeeze_excite
// FORM-COUNT-7: custom.quantized_chain = 0 : i64
// BANK-LABEL: func.func @squeeze_excite
// BANK: buckyball.bank_custom "lookup_i8"
// BANK: buckyball.bank_custom "gate_mul_i8"
// BANK: buckyball.bank_custom "matmul_i8_block"
// BANK: buckyball.bank_mvout
// BANK: return
func.func @squeeze_excite(%x: tensor<1x2x4x32xi8>, %w1: tensor<32x16xi8>, %b1: tensor<1x16xi32>, %s1: tensor<1x16xf32>, %w2: tensor<16x32xi8>, %b2: tensor<1x32xi32>, %s2: tensor<1x32xf32>, %lut: tensor<256xi8>, %w3: tensor<32x16xi8>) -> tensor<8x16xi8> {
  %source = tensor.collapse_shape %x [[0,1,2],[3]] : tensor<1x2x4x32xi8> into tensor<8x32xi8>
  %avg = custom.op "global_avg_i8"(%x) {input_scale = 1.0 : f32, output_scale = 1.0 : f32} : (tensor<1x2x4x32xi8>) -> tensor<1x1x1x32xi8>
  %vector = tensor.collapse_shape %avg [[0,1,2],[3]] : tensor<1x1x1x32xi8> into tensor<1x32xi8>
  %first = custom.op "linear_i8_requant"(%vector,%w1,%b1,%s1) {relu = true} : (tensor<1x32xi8>,tensor<32x16xi8>,tensor<1x16xi32>,tensor<1x16xf32>) -> tensor<1x16xi8>
  %second = custom.op "linear_i8_requant"(%first,%w2,%b2,%s2) {relu = false} : (tensor<1x16xi8>,tensor<16x32xi8>,tensor<1x32xi32>,tensor<1x32xf32>) -> tensor<1x32xi8>
  %gate = custom.op "lookup_i8"(%second,%lut) : (tensor<1x32xi8>,tensor<256xi8>) -> tensor<1x32xi8>
  %multiplied = custom.op "gate_mul_i8"(%gate,%source) {ratio = 0.5 : f32} : (tensor<1x32xi8>,tensor<8x32xi8>) -> tensor<8x32xi8>
  %result = custom.op "linear_i8_requant"(%multiplied,%w3,%b1,%s1) {relu = false} : (tensor<8x32xi8>,tensor<32x16xi8>,tensor<1x16xi32>,tensor<1x16xf32>) -> tensor<8x16xi8>
  return %result : tensor<8x16xi8>
}
