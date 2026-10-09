func.func @tensor_unfused(%a: tensor<8x64xf32>, %b: tensor<64x48xf32>) -> tensor<8x48xf32> {
  %0 = buckyball.fp32_matmul %a, %b <fused = false> : tensor<8x64xf32>, tensor<64x48xf32> -> tensor<8x48xf32>
  return %0 : tensor<8x48xf32>
}
func.func @tensor_fma(%a: tensor<8x64xf32>, %b: tensor<64x48xf32>) -> tensor<8x48xf32> {
  %0 = buckyball.fp32_matmul %a, %b <fused = true> : tensor<8x64xf32>, tensor<64x48xf32> -> tensor<8x48xf32>
  return %0 : tensor<8x48xf32>
}
