// An attention without a mask over 2 heads, 20 queries, 32 keys of 64 values:
// scores = q . k^T (k its 32 keys of 64 values, transposed), their softmax scaled by 1/8, the output = P . v, every
// product on MXINT8 operands quantized online. convert-linalg-to-hart makes
// it one bbacc.custom "attention" a head (OutlineAttention, SplitRows), which
// convert-linalg-to-acc computes in the thread's banks (LowerAttention).
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#row = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#one = affine_map<(d0, d1, d2) -> (0, 0, 0)>
func.func @compute(%q: memref<2x20x64xf32>, %k: memref<2x32x64xf32>, %v: memref<2x32x64xf32>, %out: memref<2x20x64xf32>) attributes {llvm.emit_c_interface} {
  %qt = bufferization.to_tensor %q restrict : memref<2x20x64xf32> to tensor<2x20x64xf32>
  %kr = bufferization.to_tensor %k restrict : memref<2x32x64xf32> to tensor<2x32x64xf32>
  %ke = tensor.empty() : tensor<2x64x32xf32>
  %kt = linalg.transpose ins(%kr : tensor<2x32x64xf32>) outs(%ke : tensor<2x64x32xf32>) permutation = [0, 2, 1]
  %vt = bufferization.to_tensor %v restrict : memref<2x32x64xf32> to tensor<2x32x64xf32>
  %qq = "frontend.fp32_to_mxint8"(%qt) {axis = 2 : i64} : (tensor<2x20x64xf32>) -> tensor<2x20x66xi8>
  %qd = "frontend.mxint8_to_fp32"(%qq) {axis = 2 : i64} : (tensor<2x20x66xi8>) -> tensor<2x20x64xf32>
  %kq = "frontend.fp32_to_mxint8"(%kt) {axis = 1 : i64} : (tensor<2x64x32xf32>) -> tensor<2x66x32xi8>
  %kd = "frontend.mxint8_to_fp32"(%kq) {axis = 1 : i64} : (tensor<2x66x32xi8>) -> tensor<2x64x32xf32>
  %zero = arith.constant 0.0 : f32
  %se = tensor.empty() : tensor<2x20x32xf32>
  %s0 = linalg.fill ins(%zero : f32) outs(%se : tensor<2x20x32xf32>) -> tensor<2x20x32xf32>
  %scores = linalg.batch_matmul ins(%qd, %kd : tensor<2x20x64xf32>, tensor<2x64x32xf32>) outs(%s0 : tensor<2x20x32xf32>) -> tensor<2x20x32xf32>
  %eighth = arith.constant dense<1.250000e-01> : tensor<1x1x1xf32>
  %scaled = tensor.empty() : tensor<2x20x32xf32>
  %xt = linalg.generic {indexing_maps = [#id, #one, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%scores, %eighth : tensor<2x20x32xf32>, tensor<1x1x1xf32>) outs(%scaled : tensor<2x20x32xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %p = arith.mulf %a, %b : f32
      linalg.yield %p : f32
  } -> tensor<2x20x32xf32>
  %lowest = arith.constant -3.40282347E+38 : f32
  %re = tensor.empty() : tensor<2x20xf32>
  %mi = linalg.fill ins(%lowest : f32) outs(%re : tensor<2x20xf32>) -> tensor<2x20xf32>
  %m = linalg.reduce ins(%xt : tensor<2x20x32xf32>) outs(%mi : tensor<2x20xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %p = arith.maximumf %in, %init : f32
      linalg.yield %p : f32
    }
  %me = tensor.expand_shape %m [[0], [1, 2]] output_shape [2, 20, 1] : tensor<2x20xf32> into tensor<2x20x1xf32>
  %e0 = tensor.empty() : tensor<2x20x32xf32>
  %d = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%xt, %me : tensor<2x20x32xf32>, tensor<2x20x1xf32>) outs(%e0 : tensor<2x20x32xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %p = arith.subf %a, %b : f32
      linalg.yield %p : f32
  } -> tensor<2x20x32xf32>
  %e = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%d : tensor<2x20x32xf32>) outs(%e0 : tensor<2x20x32xf32>) {
    ^bb0(%a: f32, %o: f32):
      %p = math.exp %a : f32
      linalg.yield %p : f32
  } -> tensor<2x20x32xf32>
  %si = linalg.fill ins(%zero : f32) outs(%re : tensor<2x20xf32>) -> tensor<2x20xf32>
  %s = linalg.reduce ins(%e : tensor<2x20x32xf32>) outs(%si : tensor<2x20xf32>) dimensions = [2]
    (%in: f32, %init: f32) {
      %p = arith.addf %in, %init : f32
      linalg.yield %p : f32
    }
  %sx = tensor.expand_shape %s [[0], [1, 2]] output_shape [2, 20, 1] : tensor<2x20xf32> into tensor<2x20x1xf32>
  %y = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%e, %sx : tensor<2x20x32xf32>, tensor<2x20x1xf32>) outs(%e0 : tensor<2x20x32xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %p = arith.divf %a, %b : f32
      linalg.yield %p : f32
  } -> tensor<2x20x32xf32>
  %pq = "frontend.fp32_to_mxint8"(%y) {axis = 2 : i64} : (tensor<2x20x32xf32>) -> tensor<2x20x33xi8>
  %pd = "frontend.mxint8_to_fp32"(%pq) {axis = 2 : i64} : (tensor<2x20x33xi8>) -> tensor<2x20x32xf32>
  %vq = "frontend.fp32_to_mxint8"(%vt) {axis = 1 : i64} : (tensor<2x32x64xf32>) -> tensor<2x33x64xi8>
  %vd = "frontend.mxint8_to_fp32"(%vq) {axis = 1 : i64} : (tensor<2x33x64xi8>) -> tensor<2x32x64xf32>
  %oe = tensor.empty() : tensor<2x20x64xf32>
  %o0 = linalg.fill ins(%zero : f32) outs(%oe : tensor<2x20x64xf32>) -> tensor<2x20x64xf32>
  %o = linalg.batch_matmul ins(%pd, %vd : tensor<2x20x32xf32>, tensor<2x32x64xf32>) outs(%o0 : tensor<2x20x64xf32>) -> tensor<2x20x64xf32>
  bufferization.materialize_in_destination %o in writable %out : (tensor<2x20x64xf32>, memref<2x20x64xf32>) -> ()
  return
}
