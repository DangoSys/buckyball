// SHM-LABEL: func.func @compute(
// SHM: call @stream_submit
#a = affine_map<(m,n,k)->(m,k)>
#w = affine_map<(m,n,k)->(n,k)>
#c = affine_map<(m,n,k)->(m,n)>
func.func @compute(%input: memref<4x33xi8>, %weights: memref<1x640x33xi8>, %out: memref<4x330xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<4x33xi8> to tensor<4x33xi8>
  %w = bufferization.to_tensor %weights restrict : memref<1x640x33xi8> to tensor<1x640x33xi8>
  %xe = tensor.empty() : tensor<4x33xi8>
  %we = tensor.empty() : tensor<1x640x33xi8>
  %sx = hart.transfer %x to(%xe) <to = 2> : (tensor<4x33xi8>, tensor<4x33xi8>) -> tensor<4x33xi8>
  %sw = hart.transfer %w to(%we) <to = 2> : (tensor<1x640x33xi8>, tensor<1x640x33xi8>) -> tensor<1x640x33xi8>
  %empty = tensor.empty() : tensor<4x330xi8>
  %q = hart.launch @quantize(%sx,%sw) outs(%empty) <thread = 2> : (tensor<4x33xi8>, tensor<1x640x33xi8>, tensor<4x330xi8>) -> tensor<4x330xi8>
  %result_empty = tensor.empty() : tensor<4x330xi8>
  %result = hart.transfer %q to(%result_empty) <from = 2> : (tensor<4x330xi8>, tensor<4x330xi8>) -> tensor<4x330xi8>
  bufferization.materialize_in_destination %result in restrict writable %out : (tensor<4x330xi8>,memref<4x330xi8>) -> ()
  return
}
func.func private @quantize(%input: tensor<4x33xi8> {bufferization.buffer_layout = affine_map<(d0,d1)->(d0,d1)>, bufferization.writable = false}, %weights: tensor<1x640x33xi8> {bufferization.buffer_layout = affine_map<(d0,d1,d2)->(d0,d1,d2)>, bufferization.writable = false}, %out: tensor<4x330xi8> {bufferization.buffer_layout = affine_map<(d0,d1)->(d0,d1)>, bufferization.writable = true}) -> tensor<4x330xi8> attributes {bbacc.target = "ffn", hart.program_bank = 40 : i64} {
  %ad = "frontend.mxint8_to_fp32"(%input) {axis = 1 : i64} : (tensor<4x33xi8>) -> tensor<4x32xf32>
  %wd = "frontend.mxint8_to_fp32"(%weights) {axis = 2 : i64} : (tensor<1x640x33xi8>) -> tensor<1x640x32xf32>
  %we = tensor.empty() : tensor<640x1x32xf32>
  %wt = linalg.transpose ins(%wd : tensor<1x640x32xf32>) outs(%we : tensor<640x1x32xf32>) permutation = [1,0,2]
  %wm = tensor.collapse_shape %wt [[0],[1,2]] : tensor<640x1x32xf32> into tensor<640x32xf32>
  %ce = tensor.empty() : tensor<4x640xf32>
  %zero = arith.constant 0.0 : f32
  %init = linalg.fill ins(%zero : f32) outs(%ce : tensor<4x640xf32>) -> tensor<4x640xf32>
  %storage = tensor.empty() : tensor<4x640xf32>
  %product = linalg.matmul indexing_maps=[#a,#w,#c] ins(%ad,%wm : tensor<4x32xf32>,tensor<640x32xf32>) outs(%init : tensor<4x640xf32>) -> tensor<4x640xf32>
  %written_product = bufferization.materialize_in_destination %product in %storage : (tensor<4x640xf32>, tensor<4x640xf32>) -> tensor<4x640xf32>
  %slice = tensor.extract_slice %written_product[0,320] [4,320] [1,1] : tensor<4x640xf32> to tensor<4x320xf32>
  %q = "frontend.fp32_to_mxint8"(%slice) {axis = 1 : i64} : (tensor<4x320xf32>) -> tensor<4x330xi8>
  %written = bufferization.materialize_in_destination %q in %out : (tensor<4x330xi8>, tensor<4x330xi8>) -> tensor<4x330xi8>
  return %written : tensor<4x330xi8>
}
