#a2 = affine_map<(a,b)->(a,b)>
#a3 = affine_map<(a,b,c)->(a,b,c)>
#ma = affine_map<(m,n,k)->(m,k)>
#mw = affine_map<(m,n,k)->(n,k)>
#mc = affine_map<(m,n,k)->(m,n)>
module {
  func.func @unsupported(%input: tensor<1x3168xi8> {bufferization.buffer_layout = #a2}, %weights: tensor<1x32x2112xi8> {bufferization.buffer_layout = #a3}, %out: tensor<1x33xi8> {bufferization.buffer_layout = #a2, bufferization.writable = true}) -> tensor<1x33xi8> attributes {bbacc.target = "mixed5"} {
    %a = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<1x3168xi8>) -> tensor<1x3072xf32>
    // expected-error @+1 {{INT5 matrix needs a supported dense matmul-to-MXINT8 boundary}}
    %w = "frontend.mxint5_to_fp32"(%weights) {axis = 2 : i64} : (tensor<1x32x2112xi8>) -> tensor<1x32x3072xf32>
    %we = tensor.empty() : tensor<32x1x3072xf32>
    %wt = linalg.transpose ins(%w : tensor<1x32x3072xf32>) outs(%we : tensor<32x1x3072xf32>) permutation = [1,0,2]
    %wc = tensor.collapse_shape %wt [[0],[1,2]] : tensor<32x1x3072xf32> into tensor<32x3072xf32>
    %ce = tensor.empty() : tensor<1x32xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<1x32xf32>) -> tensor<1x32xf32>
    %product = linalg.matmul indexing_maps=[#ma,#mw,#mc] ins(%a,%wc : tensor<1x3072xf32>,tensor<32x3072xf32>) outs(%init : tensor<1x32xf32>) -> tensor<1x32xf32>
    %packed = "frontend.fp32_to_mxint8"(%product) {axis = 1 : i64} : (tensor<1x32xf32>) -> tensor<1x33xi8>
    %written = bufferization.materialize_in_destination %packed in %out : (tensor<1x33xi8>,tensor<1x33xi8>) -> tensor<1x33xi8>
    return %written : tensor<1x33xi8>
  }
}
