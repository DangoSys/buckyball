func.func @compute(%input: memref<8x44xf32>, %output: memref<44x8xf32>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<8x44xf32> to tensor<8x44xf32>
  %empty = tensor.empty() : tensor<44x8xf32>
  %result = linalg.transpose ins(%a : tensor<8x44xf32>) outs(%empty : tensor<44x8xf32>) permutation = [1, 0]
  bufferization.materialize_in_destination %result in writable %output : (tensor<44x8xf32>, memref<44x8xf32>) -> ()
  return
}
