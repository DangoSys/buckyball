// SHM-LABEL: func.func @compute(
// SHM: call @stream_transfer
// SHM: call @stream_submit
// Five 32-row chunks: x*x+x on GP threads, MXINT8 quantization, then a
// product with a 48x256 MXINT8 weight; each chunk passes thread to thread.
#id = affine_map<(m,n)->(m,n)>
#a = affine_map<(m,n,k)->(m,k)>
#w = affine_map<(m,n,k)->(n,k)>
#c = affine_map<(m,n,k)->(m,n)>
func.func @compute(%input: memref<160x256xf32>, %weight: memref<1x48x264xi8>, %out: memref<160x48xf32>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<160x256xf32> to tensor<160x256xf32>
  %raw = bufferization.to_tensor %weight restrict : memref<1x48x264xi8> to tensor<1x48x264xi8>
  %wd = "frontend.mxint8_to_fp32"(%raw) {axis = 2 : i64} : (tensor<1x48x264xi8>) -> tensor<1x48x256xf32>
  %we = tensor.empty() : tensor<48x1x256xf32>
  %wt = linalg.transpose ins(%wd : tensor<1x48x256xf32>) outs(%we : tensor<48x1x256xf32>) permutation = [1,0,2]
  %weights = tensor.collapse_shape %wt [[0],[1,2]] : tensor<48x1x256xf32> into tensor<48x256xf32>
  %zero = arith.constant 0.0 : f32
  %acc0 = tensor.empty() : tensor<160x48xf32>
  %x0 = tensor.extract_slice %x[0, 0] [32, 256] [1, 1] : tensor<160x256xf32> to tensor<32x256xf32>
  %e0 = tensor.empty() : tensor<32x256xf32>
  %g0 = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel"]} ins(%x0 : tensor<32x256xf32>) outs(%e0 : tensor<32x256xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %sq = arith.mulf %v, %v : f32
    %value = arith.addf %sq, %v : f32
    linalg.yield %value : f32
  } -> tensor<32x256xf32>
  %q0 = "frontend.fp32_to_mxint8"(%g0) {axis = 1 : i64} : (tensor<32x256xf32>) -> tensor<32x264xi8>
  %d0 = "frontend.mxint8_to_fp32"(%q0) {axis = 1 : i64} : (tensor<32x264xi8>) -> tensor<32x256xf32>
  %ce0 = tensor.empty() : tensor<32x48xf32>
  %i0 = linalg.fill ins(%zero : f32) outs(%ce0 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %p0 = linalg.matmul indexing_maps=[#a,#w,#c] ins(%d0,%weights : tensor<32x256xf32>,tensor<48x256xf32>) outs(%i0 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %acc1 = tensor.insert_slice %p0 into %acc0[0, 0] [32, 48] [1, 1] : tensor<32x48xf32> into tensor<160x48xf32>
  %x1 = tensor.extract_slice %x[32, 0] [32, 256] [1, 1] : tensor<160x256xf32> to tensor<32x256xf32>
  %e1 = tensor.empty() : tensor<32x256xf32>
  %g1 = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel"]} ins(%x1 : tensor<32x256xf32>) outs(%e1 : tensor<32x256xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %sq = arith.mulf %v, %v : f32
    %value = arith.addf %sq, %v : f32
    linalg.yield %value : f32
  } -> tensor<32x256xf32>
  %q1 = "frontend.fp32_to_mxint8"(%g1) {axis = 1 : i64} : (tensor<32x256xf32>) -> tensor<32x264xi8>
  %d1 = "frontend.mxint8_to_fp32"(%q1) {axis = 1 : i64} : (tensor<32x264xi8>) -> tensor<32x256xf32>
  %ce1 = tensor.empty() : tensor<32x48xf32>
  %i1 = linalg.fill ins(%zero : f32) outs(%ce1 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %p1 = linalg.matmul indexing_maps=[#a,#w,#c] ins(%d1,%weights : tensor<32x256xf32>,tensor<48x256xf32>) outs(%i1 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %acc2 = tensor.insert_slice %p1 into %acc1[32, 0] [32, 48] [1, 1] : tensor<32x48xf32> into tensor<160x48xf32>
  %x2 = tensor.extract_slice %x[64, 0] [32, 256] [1, 1] : tensor<160x256xf32> to tensor<32x256xf32>
  %e2 = tensor.empty() : tensor<32x256xf32>
  %g2 = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel"]} ins(%x2 : tensor<32x256xf32>) outs(%e2 : tensor<32x256xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %sq = arith.mulf %v, %v : f32
    %value = arith.addf %sq, %v : f32
    linalg.yield %value : f32
  } -> tensor<32x256xf32>
  %q2 = "frontend.fp32_to_mxint8"(%g2) {axis = 1 : i64} : (tensor<32x256xf32>) -> tensor<32x264xi8>
  %d2 = "frontend.mxint8_to_fp32"(%q2) {axis = 1 : i64} : (tensor<32x264xi8>) -> tensor<32x256xf32>
  %ce2 = tensor.empty() : tensor<32x48xf32>
  %i2 = linalg.fill ins(%zero : f32) outs(%ce2 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %p2 = linalg.matmul indexing_maps=[#a,#w,#c] ins(%d2,%weights : tensor<32x256xf32>,tensor<48x256xf32>) outs(%i2 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %acc3 = tensor.insert_slice %p2 into %acc2[64, 0] [32, 48] [1, 1] : tensor<32x48xf32> into tensor<160x48xf32>
  %x3 = tensor.extract_slice %x[96, 0] [32, 256] [1, 1] : tensor<160x256xf32> to tensor<32x256xf32>
  %e3 = tensor.empty() : tensor<32x256xf32>
  %g3 = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel"]} ins(%x3 : tensor<32x256xf32>) outs(%e3 : tensor<32x256xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %sq = arith.mulf %v, %v : f32
    %value = arith.addf %sq, %v : f32
    linalg.yield %value : f32
  } -> tensor<32x256xf32>
  %q3 = "frontend.fp32_to_mxint8"(%g3) {axis = 1 : i64} : (tensor<32x256xf32>) -> tensor<32x264xi8>
  %d3 = "frontend.mxint8_to_fp32"(%q3) {axis = 1 : i64} : (tensor<32x264xi8>) -> tensor<32x256xf32>
  %ce3 = tensor.empty() : tensor<32x48xf32>
  %i3 = linalg.fill ins(%zero : f32) outs(%ce3 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %p3 = linalg.matmul indexing_maps=[#a,#w,#c] ins(%d3,%weights : tensor<32x256xf32>,tensor<48x256xf32>) outs(%i3 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %acc4 = tensor.insert_slice %p3 into %acc3[96, 0] [32, 48] [1, 1] : tensor<32x48xf32> into tensor<160x48xf32>
  %x4 = tensor.extract_slice %x[128, 0] [32, 256] [1, 1] : tensor<160x256xf32> to tensor<32x256xf32>
  %e4 = tensor.empty() : tensor<32x256xf32>
  %g4 = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel"]} ins(%x4 : tensor<32x256xf32>) outs(%e4 : tensor<32x256xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %sq = arith.mulf %v, %v : f32
    %value = arith.addf %sq, %v : f32
    linalg.yield %value : f32
  } -> tensor<32x256xf32>
  %q4 = "frontend.fp32_to_mxint8"(%g4) {axis = 1 : i64} : (tensor<32x256xf32>) -> tensor<32x264xi8>
  %d4 = "frontend.mxint8_to_fp32"(%q4) {axis = 1 : i64} : (tensor<32x264xi8>) -> tensor<32x256xf32>
  %ce4 = tensor.empty() : tensor<32x48xf32>
  %i4 = linalg.fill ins(%zero : f32) outs(%ce4 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %p4 = linalg.matmul indexing_maps=[#a,#w,#c] ins(%d4,%weights : tensor<32x256xf32>,tensor<48x256xf32>) outs(%i4 : tensor<32x48xf32>) -> tensor<32x48xf32>
  %acc5 = tensor.insert_slice %p4 into %acc4[128, 0] [32, 48] [1, 1] : tensor<32x48xf32> into tensor<160x48xf32>
  bufferization.materialize_in_destination %acc5 in restrict writable %out : (tensor<160x48xf32>,memref<160x48xf32>) -> ()
  return
}
