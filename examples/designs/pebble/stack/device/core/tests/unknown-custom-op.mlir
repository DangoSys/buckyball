func.func @unknown(%input: tensor<1x32xf32>) -> tensor<1x32xf32> {
  // expected-error @+1 {{has no bank lowering on this core for unknown}}
  %result = bbacc.custom "unknown"(%input) : (tensor<1x32xf32>) -> tensor<1x32xf32>
  return %result : tensor<1x32xf32>
}
