// A masked attention as a step's: 2 key heads of 3 query heads each, 8
// queries (24 rows a key head), 96 keys of 64 values in two parts (64 the
// prefix's, 32 the step's own, [1, T, H, D] each, inserted one after the
// other), the bias [1, 8, 96] of the scores the same for every head (0 for
// a key that is seen, -1e30 for one that is not). scores = q . k^T plus the
// bias, times 1/8, softmax, output = P . v, every product on MXINT8 operands
// quantized online. convert-linalg-to-hart makes it one bbacc.custom
// "attention" a key head reading the two parts and the bias where they are
// (OutlineAttention, SplitRows), which convert-linalg-to-acc computes in
// the thread's banks (LowerAttention).
#id = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#one = affine_map<(d0, d1, d2, d3, d4) -> (d0, 0, 0, 0, 0)>
#mask = affine_map<(d0, d1, d2, d3, d4) -> (d0, 0, 0, d3, d4)>
#row = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, 0)>
func.func @compute(%q: memref<2x24x64xf32>, %kp: memref<1x64x2x64xf32>, %kn: memref<1x32x2x64xf32>, %vp: memref<1x64x2x64xf32>, %vn: memref<1x32x2x64xf32>, %bias: memref<1x8x96xf32>, %out: memref<2x24x64xf32>) attributes {llvm.emit_c_interface} {
  %qt = bufferization.to_tensor %q restrict : memref<2x24x64xf32> to tensor<2x24x64xf32>
  %kpt = bufferization.to_tensor %kp restrict : memref<1x64x2x64xf32> to tensor<1x64x2x64xf32>
  %knt = bufferization.to_tensor %kn restrict : memref<1x32x2x64xf32> to tensor<1x32x2x64xf32>
  %vpt = bufferization.to_tensor %vp restrict : memref<1x64x2x64xf32> to tensor<1x64x2x64xf32>
  %vnt = bufferization.to_tensor %vn restrict : memref<1x32x2x64xf32> to tensor<1x32x2x64xf32>
  %bt = bufferization.to_tensor %bias restrict : memref<1x8x96xf32> to tensor<1x8x96xf32>
  %ce = tensor.empty() : tensor<1x96x2x64xf32>
  %k0 = tensor.insert_slice %kpt into %ce[0, 0, 0, 0] [1, 64, 2, 64] [1, 1, 1, 1] : tensor<1x64x2x64xf32> into tensor<1x96x2x64xf32>
  %kc = tensor.insert_slice %knt into %k0[0, 64, 0, 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x2x64xf32> into tensor<1x96x2x64xf32>
  %v0 = tensor.insert_slice %vpt into %ce[0, 0, 0, 0] [1, 64, 2, 64] [1, 1, 1, 1] : tensor<1x64x2x64xf32> into tensor<1x96x2x64xf32>
  %vc = tensor.insert_slice %vnt into %v0[0, 64, 0, 0] [1, 32, 2, 64] [1, 1, 1, 1] : tensor<1x32x2x64xf32> into tensor<1x96x2x64xf32>
  %he = tensor.empty() : tensor<1x2x96x64xf32>
  %kh = linalg.transpose ins(%kc : tensor<1x96x2x64xf32>) outs(%he : tensor<1x2x96x64xf32>) permutation = [0, 2, 1, 3]
  %vh = linalg.transpose ins(%vc : tensor<1x96x2x64xf32>) outs(%he : tensor<1x2x96x64xf32>) permutation = [0, 2, 1, 3]
  %te = tensor.empty() : tensor<1x2x64x96xf32>
  %kt = linalg.transpose ins(%kh : tensor<1x2x96x64xf32>) outs(%te : tensor<1x2x64x96xf32>) permutation = [0, 1, 3, 2]
  %k3 = tensor.collapse_shape %kt [[0, 1], [2], [3]] : tensor<1x2x64x96xf32> into tensor<2x64x96xf32>
  %v3 = tensor.collapse_shape %vh [[0, 1], [2], [3]] : tensor<1x2x96x64xf32> into tensor<2x96x64xf32>
  %qq = "frontend.fp32_to_mxint8"(%qt) {axis = 2 : i64} : (tensor<2x24x64xf32>) -> tensor<2x24x66xi8>
  %qd = "frontend.mxint8_to_fp32"(%qq) {axis = 2 : i64} : (tensor<2x24x66xi8>) -> tensor<2x24x64xf32>
  %kq = "frontend.fp32_to_mxint8"(%k3) {axis = 1 : i64} : (tensor<2x64x96xf32>) -> tensor<2x66x96xi8>
  %kd = "frontend.mxint8_to_fp32"(%kq) {axis = 1 : i64} : (tensor<2x66x96xi8>) -> tensor<2x64x96xf32>
  %zero = arith.constant 0.0 : f32
  %se = tensor.empty() : tensor<2x24x96xf32>
  %s0 = linalg.fill ins(%zero : f32) outs(%se : tensor<2x24x96xf32>) -> tensor<2x24x96xf32>
  %scores = linalg.batch_matmul ins(%qd, %kd : tensor<2x24x64xf32>, tensor<2x64x96xf32>) outs(%s0 : tensor<2x24x96xf32>) -> tensor<2x24x96xf32>
  %s5 = tensor.expand_shape %scores [[0, 1], [2, 3], [4]] output_shape [1, 2, 3, 8, 96] : tensor<2x24x96xf32> into tensor<1x2x3x8x96xf32>
  %b5 = tensor.expand_shape %bt [[0, 1, 2], [3], [4]] output_shape [1, 1, 1, 8, 96] : tensor<1x8x96xf32> into tensor<1x1x1x8x96xf32>
  %e5 = tensor.empty() : tensor<1x2x3x8x96xf32>
  %x5 = linalg.generic {indexing_maps = [#id, #mask, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%s5, %b5 : tensor<1x2x3x8x96xf32>, tensor<1x1x1x8x96xf32>) outs(%e5 : tensor<1x2x3x8x96xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %p = arith.addf %a, %b : f32
      linalg.yield %p : f32
  } -> tensor<1x2x3x8x96xf32>
  %eighth = arith.constant dense<1.250000e-01> : tensor<1x1x1x1x1xf32>
  %xt = linalg.generic {indexing_maps = [#id, #one, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%x5, %eighth : tensor<1x2x3x8x96xf32>, tensor<1x1x1x1x1xf32>) outs(%e5 : tensor<1x2x3x8x96xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %p = arith.mulf %a, %b : f32
      linalg.yield %p : f32
  } -> tensor<1x2x3x8x96xf32>
  %lowest = arith.constant -3.40282347E+38 : f32
  %re = tensor.empty() : tensor<1x2x3x8xf32>
  %mi = linalg.fill ins(%lowest : f32) outs(%re : tensor<1x2x3x8xf32>) -> tensor<1x2x3x8xf32>
  %m = linalg.reduce ins(%xt : tensor<1x2x3x8x96xf32>) outs(%mi : tensor<1x2x3x8xf32>) dimensions = [4]
    (%in: f32, %init: f32) {
      %p = arith.maximumf %in, %init : f32
      linalg.yield %p : f32
    }
  %me = tensor.expand_shape %m [[0], [1], [2], [3, 4]] output_shape [1, 2, 3, 8, 1] : tensor<1x2x3x8xf32> into tensor<1x2x3x8x1xf32>
  %d = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%xt, %me : tensor<1x2x3x8x96xf32>, tensor<1x2x3x8x1xf32>) outs(%e5 : tensor<1x2x3x8x96xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %p = arith.subf %a, %b : f32
      linalg.yield %p : f32
  } -> tensor<1x2x3x8x96xf32>
  %e = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%d : tensor<1x2x3x8x96xf32>) outs(%e5 : tensor<1x2x3x8x96xf32>) {
    ^bb0(%a: f32, %o: f32):
      %p = math.exp %a : f32
      linalg.yield %p : f32
  } -> tensor<1x2x3x8x96xf32>
  %si = linalg.fill ins(%zero : f32) outs(%re : tensor<1x2x3x8xf32>) -> tensor<1x2x3x8xf32>
  %s = linalg.reduce ins(%e : tensor<1x2x3x8x96xf32>) outs(%si : tensor<1x2x3x8xf32>) dimensions = [4]
    (%in: f32, %init: f32) {
      %p = arith.addf %in, %init : f32
      linalg.yield %p : f32
    }
  %sx = tensor.expand_shape %s [[0], [1], [2], [3, 4]] output_shape [1, 2, 3, 8, 1] : tensor<1x2x3x8xf32> into tensor<1x2x3x8x1xf32>
  %y = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
      ins(%e, %sx : tensor<1x2x3x8x96xf32>, tensor<1x2x3x8x1xf32>) outs(%e5 : tensor<1x2x3x8x96xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %p = arith.divf %a, %b : f32
      linalg.yield %p : f32
  } -> tensor<1x2x3x8x96xf32>
  %y3 = tensor.collapse_shape %y [[0, 1], [2, 3], [4]] : tensor<1x2x3x8x96xf32> into tensor<2x24x96xf32>
  %pq = "frontend.fp32_to_mxint8"(%y3) {axis = 2 : i64} : (tensor<2x24x96xf32>) -> tensor<2x24x99xi8>
  %pd = "frontend.mxint8_to_fp32"(%pq) {axis = 2 : i64} : (tensor<2x24x99xi8>) -> tensor<2x24x96xf32>
  %vq = "frontend.fp32_to_mxint8"(%v3) {axis = 1 : i64} : (tensor<2x96x64xf32>) -> tensor<2x99x64xi8>
  %vd = "frontend.mxint8_to_fp32"(%vq) {axis = 1 : i64} : (tensor<2x99x64xi8>) -> tensor<2x96x64xf32>
  %oe = tensor.empty() : tensor<2x24x64xf32>
  %o0 = linalg.fill ins(%zero : f32) outs(%oe : tensor<2x24x64xf32>) -> tensor<2x24x64xf32>
  %o = linalg.batch_matmul ins(%pd, %vd : tensor<2x24x96xf32>, tensor<2x96x64xf32>) outs(%o0 : tensor<2x24x64xf32>) -> tensor<2x24x64xf32>
  bufferization.materialize_in_destination %o in writable %out : (tensor<2x24x64xf32>, memref<2x24x64xf32>) -> ()
  return
}
