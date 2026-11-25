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

func.func @compute_transposed_rows(%a: memref<2x?x64xf32>, %w: memref<2x32x64xf32>, %out: memref<2x?x32xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x?x64xf32> to tensor<2x?x64xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x32x64xf32> to tensor<2x32x64xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 2 : i64} : (tensor<2x?x64xf32>) -> tensor<2x?x66xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 2 : i64} : (tensor<2x?x66xi8>) -> tensor<2x?x64xf32>
  %wq = "frontend.fp32_to_mxint8"(%wt) {axis = 2 : i64} : (tensor<2x32x64xf32>) -> tensor<2x32x66xi8>
  %wd = "frontend.mxint8_to_fp32"(%wq) {axis = 2 : i64} : (tensor<2x32x66xi8>) -> tensor<2x32x64xf32>
  %expanded = tensor.expand_shape %wd [[0], [1, 2], [3]] output_shape [2, 1, 32, 64] : tensor<2x32x64xf32> into tensor<2x1x32x64xf32>
  %empty = tensor.empty() : tensor<2x1x64x32xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x1x32x64xf32>) outs(%empty : tensor<2x1x64x32xf32>) permutation = [0, 1, 3, 2]
  %weights = tensor.collapse_shape %transposed [[0], [1, 2], [3]] : tensor<2x1x64x32xf32> into tensor<2x64x32xf32>
  %one = arith.constant 1 : index
  %m = tensor.dim %at, %one : tensor<2x?x64xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m) : tensor<2x?x32xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x32xf32>) -> tensor<2x?x32xf32>
  %product = linalg.batch_matmul ins(%ad, %weights : tensor<2x?x64xf32>, tensor<2x64x32xf32>) outs(%c : tensor<2x?x32xf32>) -> tensor<2x?x32xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x32xf32>, memref<2x?x32xf32>) -> ()
  return
}

func.func @compute_transposed_tail_rows(%a: memref<2x?x64xf32>, %w: memref<2x17x64xf32>, %out: memref<2x?x17xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x?x64xf32> to tensor<2x?x64xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x17x64xf32> to tensor<2x17x64xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 2 : i64} : (tensor<2x?x64xf32>) -> tensor<2x?x66xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 2 : i64} : (tensor<2x?x66xi8>) -> tensor<2x?x64xf32>
  %wq = "frontend.fp32_to_mxint8"(%wt) {axis = 2 : i64} : (tensor<2x17x64xf32>) -> tensor<2x17x66xi8>
  %wd = "frontend.mxint8_to_fp32"(%wq) {axis = 2 : i64} : (tensor<2x17x66xi8>) -> tensor<2x17x64xf32>
  %expanded = tensor.expand_shape %wd [[0], [1, 2], [3]] output_shape [2, 1, 17, 64] : tensor<2x17x64xf32> into tensor<2x1x17x64xf32>
  %empty = tensor.empty() : tensor<2x1x64x17xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<2x1x17x64xf32>) outs(%empty : tensor<2x1x64x17xf32>) permutation = [0, 1, 3, 2]
  %weights = tensor.collapse_shape %transposed [[0], [1, 2], [3]] : tensor<2x1x64x17xf32> into tensor<2x64x17xf32>
  %one = arith.constant 1 : index
  %m = tensor.dim %at, %one : tensor<2x?x64xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m) : tensor<2x?x17xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x17xf32>) -> tensor<2x?x17xf32>
  %product = linalg.batch_matmul ins(%ad, %weights : tensor<2x?x64xf32>, tensor<2x64x17xf32>) outs(%c : tensor<2x?x17xf32>) -> tensor<2x?x17xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x17xf32>, memref<2x?x17xf32>) -> ()
  return
}

func.func @compute_residual(%a: memref<2x?x?xf32>, %w: memref<2x?x?xf32>, %out: memref<2x?x?xf32>) attributes {llvm.emit_c_interface} {
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
  %k = tensor.dim %at, %two : tensor<2x?x?xf32>
  %re = tensor.empty(%m, %k) : tensor<2x?x?xf32>
  %residual = linalg.generic {indexing_maps = [affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>, affine_map<(b,m,k)->(b,m,k)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%at, %ad : tensor<2x?x?xf32>, tensor<2x?x?xf32>) outs(%re : tensor<2x?x?xf32>) {
    ^bb0(%original: f32, %head: f32, %unused: f32):
      %difference = arith.subf %original, %head : f32
      linalg.yield %difference : f32
  } -> tensor<2x?x?xf32>
  %tq = "frontend.fp32_to_mxint8"(%residual) {axis = 2 : i64} : (tensor<2x?x?xf32>) -> tensor<2x?x?xi8>
  %td = "frontend.mxint8_to_fp32"(%tq) {axis = 2 : i64} : (tensor<2x?x?xi8>) -> tensor<2x?x?xf32>
  %product = linalg.batch_matmul ins(%ad, %wd : tensor<2x?x?xf32>, tensor<2x?x?xf32>) outs(%c : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  %tail = linalg.batch_matmul ins(%td, %wd : tensor<2x?x?xf32>, tensor<2x?x?xf32>) outs(%c : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  %sum = linalg.generic {indexing_maps = [affine_map<(b,m,n)->(b,m,n)>, affine_map<(b,m,n)->(b,m,n)>, affine_map<(b,m,n)->(b,m,n)>], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%product, %tail : tensor<2x?x?xf32>, tensor<2x?x?xf32>) outs(%ce : tensor<2x?x?xf32>) {
    ^bb0(%head: f32, %rest: f32, %unused: f32):
      %value = arith.addf %head, %rest : f32
      linalg.yield %value : f32
  } -> tensor<2x?x?xf32>
  bufferization.materialize_in_destination %sum in writable %out : (tensor<2x?x?xf32>, memref<2x?x?xf32>) -> ()
  return
}

func.func @compute_strided(%a: memref<2x?x128xf32>, %w: memref<2x64x?xf32>, %out: memref<2x?x?xf32>) attributes {llvm.emit_c_interface} {
  %whole = bufferization.to_tensor %a restrict : memref<2x?x128xf32> to tensor<2x?x128xf32>
  %axis = arith.constant 1 : index
  %height = tensor.dim %whole, %axis : tensor<2x?x128xf32>
  %at = tensor.extract_slice %whole[0, 0, 32] [2, %height, 64] [1, 1, 1] : tensor<2x?x128xf32> to tensor<2x?x64xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x64x?xf32> to tensor<2x64x?xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 2 : i64} : (tensor<2x?x64xf32>) -> tensor<2x?x66xi8>
  %wq = "frontend.fp32_to_mxint8"(%wt) {axis = 1 : i64} : (tensor<2x64x?xf32>) -> tensor<2x66x?xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 2 : i64} : (tensor<2x?x66xi8>) -> tensor<2x?x64xf32>
  %wd = "frontend.mxint8_to_fp32"(%wq) {axis = 1 : i64} : (tensor<2x66x?xi8>) -> tensor<2x64x?xf32>
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %m = tensor.dim %at, %one : tensor<2x?x64xf32>
  %n = tensor.dim %wt, %two : tensor<2x64x?xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m, %n) : tensor<2x?x?xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  %product = linalg.batch_matmul ins(%ad, %wd : tensor<2x?x64xf32>, tensor<2x64x?xf32>) outs(%c : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x?xf32>, memref<2x?x?xf32>) -> ()
  return
}

func.func @compute_column_tiles(%a: memref<2x?x?xf32>, %w: memref<2x?x32xf32>, %out: memref<2x?x32xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x?x?xf32> to tensor<2x?x?xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x?x32xf32> to tensor<2x?x32xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 2 : i64} : (tensor<2x?x?xf32>) -> tensor<2x?x?xi8>
  %wq = "frontend.fp32_to_mxint8"(%wt) {axis = 1 : i64} : (tensor<2x?x32xf32>) -> tensor<2x?x32xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 2 : i64} : (tensor<2x?x?xi8>) -> tensor<2x?x?xf32>
  %wd = "frontend.mxint8_to_fp32"(%wq) {axis = 1 : i64} : (tensor<2x?x32xi8>) -> tensor<2x?x32xf32>
  %one = arith.constant 1 : index
  %m = tensor.dim %at, %one : tensor<2x?x?xf32>
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m) : tensor<2x?x32xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x32xf32>) -> tensor<2x?x32xf32>
  %product = linalg.batch_matmul ins(%ad, %wd : tensor<2x?x?xf32>, tensor<2x?x32xf32>) outs(%c : tensor<2x?x32xf32>) -> tensor<2x?x32xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x32xf32>, memref<2x?x32xf32>) -> ()
  return
}

func.func @compute_compact(%a: memref<2x?x64xf32>, %w: memref<2x?x66xi8>, %out: memref<2x?x?xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x?x64xf32> to tensor<2x?x64xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x?x66xi8> to tensor<2x?x66xi8>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 2 : i64} : (tensor<2x?x64xf32>) -> tensor<2x?x66xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 2 : i64} : (tensor<2x?x66xi8>) -> tensor<2x?x64xf32>
  %wd = "frontend.mxint8_to_fp32"(%wt) {axis = 2 : i64} : (tensor<2x?x66xi8>) -> tensor<2x?x64xf32>
  %one = arith.constant 1 : index
  %m = tensor.dim %at, %one : tensor<2x?x64xf32>
  %n = tensor.dim %wt, %one : tensor<2x?x66xi8>
  %we = tensor.empty(%n) : tensor<2x64x?xf32>
  %weights = linalg.transpose ins(%wd : tensor<2x?x64xf32>) outs(%we : tensor<2x64x?xf32>) permutation = [0, 2, 1]
  %zero = arith.constant 0.0 : f32
  %ce = tensor.empty(%m, %n) : tensor<2x?x?xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  %product = linalg.batch_matmul ins(%ad, %weights : tensor<2x?x64xf32>, tensor<2x64x?xf32>) outs(%c : tensor<2x?x?xf32>) -> tensor<2x?x?xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x?x?xf32>, memref<2x?x?xf32>) -> ()
  return
}
