func.func @compute(%a: memref<2x?x?xf32>, %w: memref<2x?x?xf32>, %out: memref<2x?x?xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x?x?xf32> to tensor<2x?x?xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x?x?xf32> to tensor<2x?x?xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 2 : i64} : (tensor<2x?x?xf32>) -> tensor<2x?x?xi8>
  %wq = "frontend.fp32_to_mxint8"(%wt) {axis = 1 : i64} : (tensor<2x?x?xf32>) -> tensor<2x?x?xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 2 : i64} : (tensor<2x?x?xi8>) -> tensor<2x?x?xf32>
  %wd = "frontend.mxint8_to_fp32"(%wq) {axis = 1 : i64} : (tensor<2x?x?xi8>) -> tensor<2x?x?xf32>
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %m = tensor.dim %at, %one : tensor<2x?x?xf32>
  %n = tensor.dim %wt, %two : tensor<2x?x?xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m, %n) : tensor<2x?x?xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  %product = linalg.batch_matmul ins(%ad, %wd : tensor<2x?x?xf32>, tensor<2x?x?xf32>) outs(%c : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x?xf32>, memref<2x?x?xf32>) -> ()
  return
}

func.func @compute_fp32(%a: memref<2x?x?xf32>, %w: memref<2x?x?xf32>, %out: memref<2x?x?xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x?x?xf32> to tensor<2x?x?xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x?x?xf32> to tensor<2x?x?xf32>
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %m = tensor.dim %at, %one : tensor<2x?x?xf32>
  %n = tensor.dim %wt, %two : tensor<2x?x?xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m, %n) : tensor<2x?x?xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  %product = linalg.batch_matmul ins(%at, %wt : tensor<2x?x?xf32>, tensor<2x?x?xf32>) outs(%c : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x?xf32>, memref<2x?x?xf32>) -> ()
  return
}

func.func @compute_fp32_rows(%a: memref<2x?x64xf32>, %w: memref<2x32x64xf32>, %out: memref<2x?x32xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x?x64xf32> to tensor<2x?x64xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x32x64xf32> to tensor<2x32x64xf32>
  %expanded = tensor.expand_shape %wt [[0], [1, 2], [3]] output_shape [2, 1, 32, 64] : tensor<2x32x64xf32> into tensor<2x1x32x64xf32>
  %empty = tensor.empty() : tensor<2x1x64x32xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x1x32x64xf32>) outs(%empty : tensor<2x1x64x32xf32>) permutation = [0, 1, 3, 2]
  %weights = tensor.collapse_shape %transposed [[0], [1, 2], [3]] : tensor<2x1x64x32xf32> into tensor<2x64x32xf32>
  %one = arith.constant 1 : index
  %m = tensor.dim %at, %one : tensor<2x?x64xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m) : tensor<2x?x32xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x32xf32>) -> tensor<2x?x32xf32>
  %product = linalg.batch_matmul ins(%at, %weights : tensor<2x?x64xf32>, tensor<2x64x32xf32>) outs(%c : tensor<2x?x32xf32>) -> tensor<2x?x32xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x32xf32>, memref<2x?x32xf32>) -> ()
  return
}

func.func @compute_fp32_tail_rows(%a: memref<2x?x64xf32>, %w: memref<2x17x64xf32>, %out: memref<2x?x17xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x?x64xf32> to tensor<2x?x64xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x17x64xf32> to tensor<2x17x64xf32>
  %expanded = tensor.expand_shape %wt [[0], [1, 2], [3]] output_shape [2, 1, 17, 64] : tensor<2x17x64xf32> into tensor<2x1x17x64xf32>
  %empty = tensor.empty() : tensor<2x1x64x17xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x1x17x64xf32>) outs(%empty : tensor<2x1x64x17xf32>) permutation = [0, 1, 3, 2]
  %weights = tensor.collapse_shape %transposed [[0], [1, 2], [3]] : tensor<2x1x64x17xf32> into tensor<2x64x17xf32>
  %one = arith.constant 1 : index
  %m = tensor.dim %at, %one : tensor<2x?x64xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m) : tensor<2x?x17xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x17xf32>) -> tensor<2x?x17xf32>
  %product = linalg.batch_matmul ins(%at, %weights : tensor<2x?x64xf32>, tensor<2x64x17xf32>) outs(%c : tensor<2x?x17xf32>) -> tensor<2x?x17xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x17xf32>, memref<2x?x17xf32>) -> ()
  return
}
