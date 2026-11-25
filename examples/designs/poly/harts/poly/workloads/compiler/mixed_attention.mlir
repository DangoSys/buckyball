func.func @compute(%a: memref<2x3x64xf32>, %w: memref<2x64x?xf32>, %out: memref<2x3x?xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x3x64xf32> to tensor<2x3x64xf32>
  %wt = bufferization.to_tensor %w restrict : memref<2x64x?xf32> to tensor<2x64x?xf32>
  %aq = "frontend.fp32_to_mxint8"(%at) {axis = 2 : i64} : (tensor<2x3x64xf32>) -> tensor<2x3x66xi8>
  %ad = "frontend.mxint8_to_fp32"(%aq) {axis = 2 : i64} : (tensor<2x3x66xi8>) -> tensor<2x3x64xf32>
  %wq = "frontend.fp32_to_mxint8"(%wt) {axis = 1 : i64} : (tensor<2x64x?xf32>) -> tensor<2x66x?xi8>
  %wd = "frontend.mxint8_to_fp32"(%wq) {axis = 1 : i64} : (tensor<2x66x?xi8>) -> tensor<2x64x?xf32>
  %zero = arith.constant 0.0 : f32
  %two = arith.constant 2 : index
  %n = tensor.dim %wt, %two : tensor<2x64x?xf32>
  %ce = tensor.empty(%n) : tensor<2x3x?xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
  %product = linalg.batch_matmul ins(%ad, %wd : tensor<2x3x64xf32>, tensor<2x64x?xf32>) outs(%c : tensor<2x3x?xf32>) -> tensor<2x3x?xf32>
  bufferization.materialize_in_destination %product in writable %out : (tensor<2x3x?xf32>, memref<2x3x?xf32>) -> ()
  return
}

#unit_batch = affine_map<(b, k, n) -> (0, k, n)>
#identity = affine_map<(b, k, n) -> (b, k, n)>
func.func @quantized_views(%p: memref<1x1x64xf32>, %v: memref<1x64x16xf32>, %out: memref<1x1x16xf32>) attributes {llvm.emit_c_interface} {
  %pt = bufferization.to_tensor %p restrict : memref<1x1x64xf32> to tensor<1x1x64xf32>
  %vt = bufferization.to_tensor %v restrict : memref<1x64x16xf32> to tensor<1x64x16xf32>
  %pq = "frontend.fp32_to_mxint8"(%pt) {axis = 2 : i64} : (tensor<1x1x64xf32>) -> tensor<1x1x66xi8>
  %pd = "frontend.mxint8_to_fp32"(%pq) {axis = 2 : i64} : (tensor<1x1x66xi8>) -> tensor<1x1x64xf32>
  %vq = "frontend.fp32_to_mxint8"(%vt) {axis = 1 : i64} : (tensor<1x64x16xf32>) -> tensor<1x66x16xi8>
  %vd = "frontend.mxint8_to_fp32"(%vq) {axis = 1 : i64} : (tensor<1x66x16xi8>) -> tensor<1x64x16xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x64x16xf32>
  %zeros = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x64x16xf32>) -> tensor<1x64x16xf32>
  %view = linalg.generic {indexing_maps = [#unit_batch, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%vd, %zeros : tensor<1x64x16xf32>, tensor<1x64x16xf32>) outs(%empty : tensor<1x64x16xf32>) {
    ^bb0(%value: f32, %z: f32, %unused: f32):
      %added = arith.addf %value, %z : f32
      linalg.yield %added : f32
  } -> tensor<1x64x16xf32>
  %ce = tensor.empty() : tensor<1x1x16xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<1x1x16xf32>) -> tensor<1x1x16xf32>
  %result = linalg.batch_matmul ins(%pd, %view : tensor<1x1x64xf32>, tensor<1x64x16xf32>) outs(%c : tensor<1x1x16xf32>) -> tensor<1x1x16xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<1x1x16xf32>, memref<1x1x16xf32>) -> ()
  return
}

func.func @quantized_single(%p: memref<1x1x1xf32>, %v: memref<1x1x16xf32>, %out: memref<1x1x16xf32>) attributes {llvm.emit_c_interface} {
  %pt = bufferization.to_tensor %p restrict : memref<1x1x1xf32> to tensor<1x1x1xf32>
  %vt = bufferization.to_tensor %v restrict : memref<1x1x16xf32> to tensor<1x1x16xf32>
  %pq = "frontend.fp32_to_mxint8"(%pt) {axis = 2 : i64} : (tensor<1x1x1xf32>) -> tensor<1x1x33xi8>
  %pd = "frontend.mxint8_to_fp32"(%pq) {axis = 2 : i64} : (tensor<1x1x33xi8>) -> tensor<1x1x32xf32>
  %vq = "frontend.fp32_to_mxint8"(%vt) {axis = 1 : i64} : (tensor<1x1x16xf32>) -> tensor<1x33x16xi8>
  %vd = "frontend.mxint8_to_fp32"(%vq) {axis = 1 : i64} : (tensor<1x33x16xi8>) -> tensor<1x32x16xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1x16xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1x16xf32>) -> tensor<1x1x16xf32>
  %result = linalg.batch_matmul ins(%pd, %vd : tensor<1x1x32xf32>, tensor<1x32x16xf32>) outs(%init : tensor<1x1x16xf32>) -> tensor<1x1x16xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<1x1x16xf32>, memref<1x1x16xf32>) -> ()
  return
}

func.func @zero_transpose(%input: memref<1x2x4xf32>, %out: memref<1x4x2xf32>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<1x2x4xf32> to tensor<1x2x4xf32>
  %empty = tensor.empty() : tensor<1x4x2xf32>
  %transposed = linalg.transpose ins(%x : tensor<1x2x4xf32>) outs(%empty : tensor<1x4x2xf32>) permutation = [0, 2, 1]
  %zero = arith.constant 0.0 : f32
  %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%transposed : tensor<1x4x2xf32>) outs(%empty : tensor<1x4x2xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %added = arith.addf %value, %zero : f32
      linalg.yield %added : f32
  } -> tensor<1x4x2xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<1x4x2xf32>, memref<1x4x2xf32>) -> ()
  return
}

func.func @mx_shapes(%input: memref<1x?x?xf32>, %out: memref<1x1x3xi64>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<1x?x?xf32> to tensor<1x?x?xf32>
  %q = "frontend.fp32_to_mxint8"(%x) {axis = 1 : i64} : (tensor<1x?x?xf32>) -> tensor<1x?x?xi8>
  %d = "frontend.mxint8_to_fp32"(%q) {axis = 1 : i64} : (tensor<1x?x?xi8>) -> tensor<1x?x?xf32>
  %flat = tensor.collapse_shape %d [[0, 1, 2]] : tensor<1x?x?xf32> into tensor<?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %packed = tensor.dim %q, %c1 : tensor<1x?x?xi8>
  %elements = tensor.dim %flat, %c0 : tensor<?xf32>
  %columns = tensor.dim %d, %c2 : tensor<1x?x?xf32>
  %p = arith.index_cast %packed : index to i64
  %e = arith.index_cast %elements : index to i64
  %n = arith.index_cast %columns : index to i64
  memref.store %p, %out[%c0, %c0, %c0] : memref<1x1x3xi64>
  memref.store %e, %out[%c0, %c0, %c1] : memref<1x1x3xi64>
  memref.store %n, %out[%c0, %c0, %c2] : memref<1x1x3xi64>
  return
}

func.func @plain_zero_add(%input: memref<1x1x4xf32>, %out: memref<1x1x4xf32>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<1x1x4xf32> to tensor<1x1x4xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1x4xf32>
  %result = linalg.generic {indexing_maps = [#identity, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%x : tensor<1x1x4xf32>) outs(%empty : tensor<1x1x4xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %added = arith.addf %value, %zero : f32
      linalg.yield %added : f32
  } -> tensor<1x1x4xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<1x1x4xf32>, memref<1x1x4xf32>) -> ()
  return
}

func.func @quantized_views_dynamic(%p: memref<1x1x?xf32>, %v: memref<1x?x16xf32>, %out: memref<1x1x16xf32>) attributes {llvm.emit_c_interface} {
  %pt = bufferization.to_tensor %p restrict : memref<1x1x?xf32> to tensor<1x1x?xf32>
  %vt = bufferization.to_tensor %v restrict : memref<1x?x16xf32> to tensor<1x?x16xf32>
  %pq = "frontend.fp32_to_mxint8"(%pt) {axis = 2 : i64} : (tensor<1x1x?xf32>) -> tensor<1x1x?xi8>
  %pd = "frontend.mxint8_to_fp32"(%pq) {axis = 2 : i64} : (tensor<1x1x?xi8>) -> tensor<1x1x?xf32>
  %vq = "frontend.fp32_to_mxint8"(%vt) {axis = 1 : i64} : (tensor<1x?x16xf32>) -> tensor<1x?x16xi8>
  %vd = "frontend.mxint8_to_fp32"(%vq) {axis = 1 : i64} : (tensor<1x?x16xi8>) -> tensor<1x?x16xf32>
  %zero = arith.constant 0.0 : f32
  %zero_index = arith.constant 0 : index
  %sixteen = arith.constant 16 : index
  %flat = tensor.collapse_shape %vd [[0, 1, 2]] : tensor<1x?x16xf32> into tensor<?xf32>
  %elements = tensor.dim %flat, %zero_index : tensor<?xf32>
  %length = arith.divui %elements, %sixteen : index
  %expanded = tensor.expand_shape %vd [[0], [1], [2, 3]] output_shape [1, %length, 1, 16] : tensor<1x?x16xf32> into tensor<1x?x1x16xf32>
  %back = tensor.collapse_shape %expanded [[0], [1, 2], [3]] : tensor<1x?x1x16xf32> into tensor<1x?x16xf32>
  %empty = tensor.empty(%length) : tensor<1x?x16xf32>
  %zeros = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x?x16xf32>) -> tensor<1x?x16xf32>
  %view = linalg.generic {indexing_maps = [#unit_batch, #identity, #identity], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%back, %zeros : tensor<1x?x16xf32>, tensor<1x?x16xf32>) outs(%empty : tensor<1x?x16xf32>) {
    ^bb0(%value: f32, %z: f32, %unused: f32):
      %added = arith.addf %value, %z : f32
      linalg.yield %added : f32
  } -> tensor<1x?x16xf32>
  %ce = tensor.empty() : tensor<1x1x16xf32>
  %c = linalg.fill ins(%zero : f32) outs(%ce : tensor<1x1x16xf32>) -> tensor<1x1x16xf32>
  %result = linalg.batch_matmul ins(%pd, %view : tensor<1x1x?xf32>, tensor<1x?x16xf32>) outs(%c : tensor<1x1x16xf32>) -> tensor<1x1x16xf32>
  bufferization.materialize_in_destination %result in writable %out : (tensor<1x1x16xf32>, memref<1x1x16xf32>) -> ()
  return
}
