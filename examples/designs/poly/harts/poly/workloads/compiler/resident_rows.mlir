func.func @compute(%a: memref<?x1056xi8>, %w: memref<64x1056xi8>, %out: memref<?x64xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<?x1056xi8> to tensor<?x1056xi8>
  %wt = bufferization.to_tensor %w restrict : memref<64x1056xi8> to tensor<64x1056xi8>
  %ad = "frontend.mxint8_to_fp32"(%at) {axis = 1 : i64} : (tensor<?x1056xi8>) -> tensor<?x1024xf32>
  %wd = "frontend.mxint8_to_fp32"(%wt) {axis = 1 : i64} : (tensor<64x1056xi8>) -> tensor<64x1024xf32>
  %we = tensor.empty() : tensor<1024x64xf32>
  %b = linalg.transpose ins(%wd : tensor<64x1024xf32>) outs(%we : tensor<1024x64xf32>) permutation = [1, 0]
  %c0 = arith.constant 0 : index
  %rows = tensor.dim %at, %c0 : tensor<?x1056xi8>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty(%rows) : tensor<?x64xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x64xf32>) -> tensor<?x64xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<?x1024xf32>, tensor<1024x64xf32>) outs(%init : tensor<?x64xf32>) -> tensor<?x64xf32>
  bufferization.materialize_in_destination %product in restrict writable %out : (tensor<?x64xf32>, memref<?x64xf32>) -> ()
  return
}
