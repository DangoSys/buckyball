#a2 = affine_map<(a,b)->(a,b)>
#a3 = affine_map<(a,b,c)->(a,b,c)>
// CHECK-COUNT-2: mxint5gemv_ball.bank_gemv_mxint5
// CHECK-NOT: linalg.matmul
// CHECK-NOT: frontend.fp32_to_mxint8
#ma = affine_map<(m,n,k)->(m,k)>
#mw = affine_map<(m,n,k)->(n,k)>
#mc = affine_map<(m,n,k)->(m,n)>
module {
  func.func private @gemv5(%input: tensor<?x1056xi8> {bufferization.buffer_layout = #a2}, %weights: tensor<2x64x352xi8> {bufferization.buffer_layout = #a3}, %out: tensor<?x66xi8> {bufferization.buffer_layout = #a2, bufferization.writable = true}) -> tensor<?x66xi8> attributes {bbacc.target = "mixed5", hart.program_bank = 40 : i64} {
    %c0 = arith.constant 0 : index
    %rows = tensor.dim %input, %c0 : tensor<?x1056xi8>
    %a = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<?x1056xi8>) -> tensor<?x1024xf32>
    %w = "frontend.mxint5_to_fp32"(%weights) {axis = 2 : i64} : (tensor<2x64x352xi8>) -> tensor<2x64x512xf32>
    %we = tensor.empty() : tensor<64x2x512xf32>
    %wt = linalg.transpose ins(%w : tensor<2x64x512xf32>) outs(%we : tensor<64x2x512xf32>) permutation = [1,0,2]
    %wc = tensor.collapse_shape %wt [[0],[1,2]] : tensor<64x2x512xf32> into tensor<64x1024xf32>
    %ce = tensor.empty(%rows) : tensor<?x64xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<?x64xf32>) -> tensor<?x64xf32>
    %product = linalg.matmul indexing_maps=[#ma,#mw,#mc] ins(%a,%wc : tensor<?x1024xf32>,tensor<64x1024xf32>) outs(%init : tensor<?x64xf32>) -> tensor<?x64xf32>
    %packed = "frontend.fp32_to_mxint8"(%product) {axis = 1 : i64} : (tensor<?x64xf32>) -> tensor<?x66xi8>
    %written = bufferization.materialize_in_destination %packed in %out : (tensor<?x66xi8>,tensor<?x66xi8>) -> tensor<?x66xi8>
    return %written : tensor<?x66xi8>
  }
  func.func @compute(%a: memref<?x1056xi8>, %w: memref<2x64x352xi8>, %o: memref<?x66xi8>) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %rows = memref.dim %a, %c0 : memref<?x1056xi8>
    %at = bufferization.to_tensor %a restrict : memref<?x1056xi8> to tensor<?x1056xi8>
    %wt = bufferization.to_tensor %w restrict : memref<2x64x352xi8> to tensor<2x64x352xi8>
    %ot = bufferization.to_tensor %o restrict writable : memref<?x66xi8> to tensor<?x66xi8>
    %ae = tensor.empty(%rows) : tensor<?x1056xi8>
    %we = tensor.empty() : tensor<2x64x352xi8>
    %ai = hart.transfer %at to(%ae) <to = 0> : (tensor<?x1056xi8>,tensor<?x1056xi8>) -> tensor<?x1056xi8>
    %wi = hart.transfer %wt to(%we) <to = 0> : (tensor<2x64x352xi8>,tensor<2x64x352xi8>) -> tensor<2x64x352xi8>
    %oe = tensor.empty(%rows) : tensor<?x66xi8>
    %r = hart.launch @gemv5(%ai, %wi) outs(%oe) <thread = 0> : (tensor<?x1056xi8>,tensor<2x64x352xi8>,tensor<?x66xi8>) -> tensor<?x66xi8>
    %back = hart.transfer %r to(%ot) <from = 0> : (tensor<?x66xi8>,tensor<?x66xi8>) -> tensor<?x66xi8>
    bufferization.materialize_in_destination %back in restrict writable %o : (tensor<?x66xi8>,memref<?x66xi8>) -> ()
    return
  }
}
