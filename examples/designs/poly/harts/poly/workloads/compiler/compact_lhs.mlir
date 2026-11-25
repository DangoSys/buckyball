func.func @compute_compact_lhs(%a: memref<2x?x528xi8>, %w: memref<2x?x528xi8>, %out: memref<2x?x?xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x?x528xi8> to tensor<2x?x528xi8>
  %wt = bufferization.to_tensor %w restrict : memref<2x?x528xi8> to tensor<2x?x528xi8>
  %ad = "frontend.mxint8_to_fp32"(%at) {axis = 2 : i64} : (tensor<2x?x528xi8>) -> tensor<2x?x512xf32>
  %wd = "frontend.mxint8_to_fp32"(%wt) {axis = 2 : i64} : (tensor<2x?x528xi8>) -> tensor<2x?x512xf32>
  %one = arith.constant 1 : index
  %m = tensor.dim %at, %one : tensor<2x?x528xi8>
  %n = tensor.dim %wt, %one : tensor<2x?x528xi8>
  %we = tensor.empty(%n) : tensor<2x512x?xf32>
  %weights = linalg.transpose ins(%wd : tensor<2x?x512xf32>) outs(%we : tensor<2x512x?xf32>) permutation = [0, 2, 1]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m, %n) : tensor<2x?x?xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  %product = linalg.batch_matmul ins(%ad, %weights : tensor<2x?x512xf32>, tensor<2x512x?xf32>) outs(%c : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x?xf32>, memref<2x?x?xf32>) -> ()
  return
}


func.func @compute_rows(%a: tensor<?x528xi8>, %w: tensor<?x528xi8>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {llvm.emit_c_interface} {
  %ad = "frontend.mxint8_to_fp32"(%a) {axis = 1 : i64} : (tensor<?x528xi8>) -> tensor<?x512xf32>
  %wd = "frontend.mxint8_to_fp32"(%w) {axis = 1 : i64} : (tensor<?x528xi8>) -> tensor<?x512xf32>
  %zero = arith.constant 0 : index
  %m = tensor.dim %a, %zero : tensor<?x528xi8>
  %n = tensor.dim %w, %zero : tensor<?x528xi8>
  %we = tensor.empty(%n) : tensor<512x?xf32>
  %weights = linalg.transpose ins(%wd : tensor<?x512xf32>) outs(%we : tensor<512x?xf32>) permutation = [1, 0]
  %z = arith.constant 0.0 : f32
  %ce = tensor.empty(%m, %n) : tensor<?x?xf32>
  %c = linalg.fill ins(%z : f32) outs(%ce : tensor<?x?xf32>) -> tensor<?x?xf32>
  %product = linalg.matmul ins(%ad, %weights : tensor<?x512xf32>, tensor<512x?xf32>) outs(%c : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = bufferization.materialize_in_destination %product in %out : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
