func.func @compute(%a: memref<17x2048xf32>, %w: memref<2048x17xf32>, %out: memref<17x17xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<17x2048xf32> to tensor<17x2048xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2048x17xf32> to tensor<2048x17xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<17x17xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<17x17xf32>) -> tensor<17x17xf32>
  %product = linalg.matmul ins(%at, %wt : tensor<17x2048xf32>, tensor<2048x17xf32>) outs(%init : tensor<17x17xf32>) -> tensor<17x17xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<17x17xf32>, memref<17x17xf32>) -> ()
  return
}
