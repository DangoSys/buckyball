func.func @compute(%input: memref<2x3x8xf32>, %out: memref<3x2x8xf32>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<2x3x8xf32> to tensor<2x3x8xf32>
  %empty = tensor.empty() : tensor<3x2x8xf32>
  %result = linalg.transpose ins(%a : tensor<2x3x8xf32>) outs(%empty : tensor<3x2x8xf32>) permutation = [1, 0, 2]
  bufferization.materialize_in_destination %result in writable %out : (tensor<3x2x8xf32>, memref<3x2x8xf32>) -> ()
  return
}
