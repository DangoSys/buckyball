// The weight in the Ball layout: rows of 16 * S(768) = 800 bytes (768
// codes, 24 exponents, 8 zeros), 20 rows (19 padded to a multiple of 4); the
// slice drops the padding before the decoding.
func.func @compute(%a: memref<3x768xf32>, %w: memref<20x800xi8>, %out: memref<3x19xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<3x768xf32> to tensor<3x768xf32>
  %wp = bufferization.to_tensor %w restrict : memref<20x800xi8> to tensor<20x800xi8>
  %wt = tensor.extract_slice %wp[0, 0] [19, 792] [1, 1] : tensor<20x800xi8> to tensor<19x792xi8>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 1 : i64} : (tensor<3x768xf32>) -> tensor<3x792xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 1 : i64} : (tensor<3x792xi8>) -> tensor<3x768xf32>
  %wd = "frontend.mxint8_to_fp32"(%wt) {axis = 1 : i64} : (tensor<19x792xi8>) -> tensor<19x768xf32>
  %we = tensor.empty() : tensor<768x19xf32>
  %b = linalg.transpose ins(%wd : tensor<19x768xf32>) outs(%we : tensor<768x19xf32>) permutation = [1, 0]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty() : tensor<3x19xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<3x19xf32>) -> tensor<3x19xf32>
  %product = linalg.matmul ins(%ad, %b : tensor<3x768xf32>, tensor<768x19xf32>) outs(%c : tensor<3x19xf32>) -> tensor<3x19xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<3x19xf32>, memref<3x19xf32>) -> ()
  return
}
