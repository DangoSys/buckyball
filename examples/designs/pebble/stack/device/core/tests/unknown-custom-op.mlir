func.func @unknown(%input: tensor<1x32xf32>) -> tensor<1x32xf32> {
  // expected-error @+1 {{has no bank lowering on this core for unknown}}
  %result = custom.op "unknown"(%input) : (tensor<1x32xf32>) -> tensor<1x32xf32>
  return %result : tensor<1x32xf32>
}
