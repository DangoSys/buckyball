// HART-LABEL: func.func private @project(
// HART: hart.launch @project_ht0{{.*}}thread = 0
// HART: hart.launch @project_ht1{{.*}}thread = 1
// HART: hart.launch @project_ht2{{.*}}thread = 2
// HART: hart.launch @project_ht3{{.*}}thread = 3
// HART: hart.launch @project_ht4{{.*}}thread = 4
// HART-NOT: hart.launch
// HART: return
// HART-COUNT-5: bufferization.buffer_layout = strided<[165, 1], offset: ?>
// COPIES-LABEL: func.func @compute(
// COPIES-NOT: memref.copy
// COPIES-NOT: memref.load
// COPIES: return
#ma = affine_map<(m,n,k)->(m,k)>
#mw = affine_map<(m,n,k)->(n,k)>
#mc = affine_map<(m,n,k)->(m,n)>
module {
  func.func private @project(%input: tensor<?x1056xi8>, %weights: tensor<2x160x352xi8>, %out: tensor<?x165xi8> {bufferization.writable = true}) -> tensor<?x165xi8> {
    %c0 = arith.constant 0 : index
    %rows = tensor.dim %input, %c0 : tensor<?x1056xi8>
    %a = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<?x1056xi8>) -> tensor<?x1024xf32>
    %w = "frontend.mxint5_to_fp32"(%weights) {axis = 2 : i64} : (tensor<2x160x352xi8>) -> tensor<2x160x512xf32>
    %we = tensor.empty() : tensor<160x2x512xf32>
    %wt = linalg.transpose ins(%w : tensor<2x160x512xf32>) outs(%we : tensor<160x2x512xf32>) permutation = [1,0,2]
    %wc = tensor.collapse_shape %wt [[0],[1,2]] : tensor<160x2x512xf32> into tensor<160x1024xf32>
    %ce = tensor.empty(%rows) : tensor<?x160xf32>
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<?x160xf32>) -> tensor<?x160xf32>
    %product = linalg.matmul indexing_maps=[#ma,#mw,#mc] ins(%a,%wc : tensor<?x1024xf32>,tensor<160x1024xf32>) outs(%init : tensor<?x160xf32>) -> tensor<?x160xf32>
    %packed = "frontend.fp32_to_mxint8"(%product) {axis = 1 : i64} : (tensor<?x160xf32>) -> tensor<?x165xi8>
    %written = bufferization.materialize_in_destination %packed in %out : (tensor<?x165xi8>,tensor<?x165xi8>) -> tensor<?x165xi8>
    return %written : tensor<?x165xi8>
  }
  func.func @compute(%a: memref<?x1056xi8>, %w: memref<2x160x352xi8>, %o: memref<?x165xi8>) attributes {llvm.emit_c_interface} {
    %at = bufferization.to_tensor %a restrict : memref<?x1056xi8> to tensor<?x1056xi8>
    %wt = bufferization.to_tensor %w restrict : memref<2x160x352xi8> to tensor<2x160x352xi8>
    %ot = bufferization.to_tensor %o restrict writable : memref<?x165xi8> to tensor<?x165xi8>
    %r = func.call @project(%at,%wt,%ot) : (tensor<?x1056xi8>,tensor<2x160x352xi8>,tensor<?x165xi8>) -> tensor<?x165xi8>
    bufferization.materialize_in_destination %r in restrict writable %o : (tensor<?x165xi8>,memref<?x165xi8>) -> ()
    return
  }
}
