// On a GPU target (gpu=true) the host trace runtime cannot read a tensor.
func.func @gpu_tensor_trace(%value: tensor<2xf32>) -> tensor<2xf32> {
  // expected-error @+1 {{traces a tensor in GPU memory, which the host trace runtime cannot read}}
  %traced = frontend.trace_end %value {cycle = true, id = 5 : i64, id_path = [5], tensor = true} : tensor<2xf32>
  return %traced : tensor<2xf32>
}
