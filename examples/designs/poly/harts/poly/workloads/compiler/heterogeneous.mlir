// SHM-LABEL: func.func @compute(
// SHM: call @stream_transfer
// SHM: call @stream_submit
#id = affine_map<(m,n)->(m,n)>
#a = affine_map<(m,n,k)->(m,k)>
#w = affine_map<(m,n,k)->(n,k)>
#c = affine_map<(m,n,k)->(m,n)>
func.func @compute(%input: memref<4x512xf32>, %weight: memref<1x16x528xi8>, %out: memref<4x16xf32>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<4x512xf32> to tensor<4x512xf32>
  %raw = bufferization.to_tensor %weight restrict : memref<1x16x528xi8> to tensor<1x16x528xi8>
  %empty = tensor.empty() : tensor<4x512xf32>
  %twice = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel"]} ins(%x : tensor<4x512xf32>) outs(%empty : tensor<4x512xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %value = arith.addf %v, %v : f32
    linalg.yield %value : f32
  } -> tensor<4x512xf32>
  %q = "frontend.fp32_to_mxint8"(%twice) {axis = 1 : i64} : (tensor<4x512xf32>) -> tensor<4x528xi8>
  %ad = "frontend.mxint8_to_fp32"(%q) {axis = 1 : i64} : (tensor<4x528xi8>) -> tensor<4x512xf32>
  %wd = "frontend.mxint8_to_fp32"(%raw) {axis = 2 : i64} : (tensor<1x16x528xi8>) -> tensor<1x16x512xf32>
  %we = tensor.empty() : tensor<16x1x512xf32>
  %wt = linalg.transpose ins(%wd : tensor<1x16x512xf32>) outs(%we : tensor<16x1x512xf32>) permutation = [1,0,2]
  %weights = tensor.collapse_shape %wt [[0],[1,2]] : tensor<16x1x512xf32> into tensor<16x512xf32>
  %ce = tensor.empty() : tensor<4x16xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<4x16xf32>) -> tensor<4x16xf32>
  %product = linalg.matmul indexing_maps=[#a,#w,#c] ins(%ad,%weights : tensor<4x512xf32>,tensor<16x512xf32>) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %oe = tensor.empty() : tensor<4x16xf32>
  %result = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel"]} ins(%product : tensor<4x16xf32>) outs(%oe : tensor<4x16xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %value = arith.addf %v, %v : f32
    linalg.yield %value : f32
  } -> tensor<4x16xf32>
  bufferization.materialize_in_destination %result in restrict writable %out : (tensor<4x16xf32>,memref<4x16xf32>) -> ()
  return
}
