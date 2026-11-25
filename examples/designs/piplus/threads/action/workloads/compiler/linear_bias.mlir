// A linear layer, C = A . W^T + bias[1, N] (the frontend's matmul from zero
// and the broadcast add): convert-linalg-to-hart starts the product at the
// broadcast bias, and mxint8mm adds the bias itself (bias mode), N = 36 a
// tile and a part.
func.func @compute(%a: memref<3x768xf32>, %w: memref<36x768xf32>, %bias: memref<1x36xf32>, %out: memref<3x36xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<3x768xf32> to tensor<3x768xf32>
  %wt = bufferization.to_tensor %w restrict : memref<36x768xf32> to tensor<36x768xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 1 : i64} : (tensor<3x768xf32>) -> tensor<3x792xi8>
  %wq = "frontend.fp32_to_mxint8"(%wt) {axis = 1 : i64} : (tensor<36x768xf32>) -> tensor<36x792xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 1 : i64} : (tensor<3x792xi8>) -> tensor<3x768xf32>
  %wd = "frontend.mxint8_to_fp32"(%wq) {axis = 1 : i64} : (tensor<36x792xi8>) -> tensor<36x768xf32>
  %we = tensor.empty() : tensor<768x36xf32>
  %b = linalg.transpose ins(%wd : tensor<36x768xf32>) outs(%we : tensor<768x36xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<3x36xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<3x36xf32>) -> tensor<3x36xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<3x768xf32>, tensor<768x36xf32>) outs(%c : tensor<3x36xf32>) -> tensor<3x36xf32>
  %bt = bufferization.to_tensor %bias restrict : memref<1x36xf32> to tensor<1x36xf32>
  %biased = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%bt, %product : tensor<1x36xf32>, tensor<3x36xf32>) outs(%ce : tensor<3x36xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %v = arith.addf %x, %y : f32
      linalg.yield %v : f32
  } -> tensor<3x36xf32>
  bufferization.materialize_in_destination %biased in writable %out : (tensor<3x36xf32>, memref<3x36xf32>) -> ()
  return
}
