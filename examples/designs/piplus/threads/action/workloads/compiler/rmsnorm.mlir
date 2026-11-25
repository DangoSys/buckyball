// RMSNorm as the language model's decomposition over the rows of [20, 960]
// (x ^ 2 by fpowi, the row sum, 1/w, eps, rsqrt, x * r, gamma * that):
// convert-linalg-to-hart replaces it by a bbacc.custom "layernorm" with rms
// on LayerNormBall, gamma and eps in its parameter bank.
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0, 0)>
#col = affine_map<(d0, d1) -> (0, d1)>
#one = affine_map<(d0, d1) -> (0, 0)>
func.func @compute(%x: memref<20x960xf32>, %g: memref<960xf32>, %out: memref<20x960xf32>) attributes {llvm.emit_c_interface} {
  %xt = bufferization.to_tensor %x restrict : memref<20x960xf32> to tensor<20x960xf32>
  %gt = bufferization.to_tensor %g restrict : memref<960xf32> to tensor<960xf32>
  %zero = arith.constant 0.0 : f32
  %two = arith.constant 2 : i32
  %inv = arith.constant dense<0.00104166672> : tensor<1x1xf32>
  %eps = arith.constant dense<9.99999974E-6> : tensor<20x1xf32>
  %re = tensor.empty() : tensor<20xf32>
  %ri = linalg.fill ins(%zero : f32) outs(%re : tensor<20xf32>) -> tensor<20xf32>
  %e1 = tensor.empty() : tensor<20x1xf32>
  %e0 = tensor.empty() : tensor<20x960xf32>
  %sq = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%xt : tensor<20x960xf32>) outs(%e0 : tensor<20x960xf32>) {
    ^bb0(%a: f32, %o: f32):
      %v = math.fpowi %a, %two : f32, i32
      linalg.yield %v : f32
  } -> tensor<20x960xf32>
  %q = linalg.reduce ins(%sq : tensor<20x960xf32>) outs(%ri : tensor<20xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
      %v = arith.addf %in, %init : f32
      linalg.yield %v : f32
    }
  %qe = tensor.expand_shape %q [[0, 1]] output_shape [20, 1] : tensor<20xf32> into tensor<20x1xf32>
  %var = linalg.generic {indexing_maps = [#one, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%inv, %qe : tensor<1x1xf32>, tensor<20x1xf32>) outs(%e1 : tensor<20x1xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x1xf32>
  %ve = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%var, %eps : tensor<20x1xf32>, tensor<20x1xf32>) outs(%e1 : tensor<20x1xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.addf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x1xf32>
  %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%ve : tensor<20x1xf32>) outs(%e1 : tensor<20x1xf32>) {
    ^bb0(%a: f32, %o: f32):
      %v = math.rsqrt %a : f32
      linalg.yield %v : f32
  } -> tensor<20x1xf32>
  %n = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]}
      ins(%xt, %r : tensor<20x960xf32>, tensor<20x1xf32>) outs(%e0 : tensor<20x960xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x960xf32>
  %ge = tensor.expand_shape %gt [[0, 1]] output_shape [1, 960] : tensor<960xf32> into tensor<1x960xf32>
  %y = linalg.generic {indexing_maps = [#col, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%ge, %n : tensor<1x960xf32>, tensor<20x960xf32>) outs(%e0 : tensor<20x960xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %v = arith.mulf %a, %b : f32
      linalg.yield %v : f32
  } -> tensor<20x960xf32>
  bufferization.materialize_in_destination %y in writable %out : (tensor<20x960xf32>, memref<20x960xf32>) -> ()
  return
}
