// SHM-LABEL: func.func @compute(
// SHM: call @stream_submit
func.func @compute(%input: memref<2x3x5x128xf32>, %out: memref<2x2x2x132xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<2x3x5x128xf32> to tensor<2x3x5x128xf32>
  %xe = tensor.empty() : tensor<2x3x5x128xf32>
  %shared = hart.transfer %x to(%xe) <to = 2> : (tensor<2x3x5x128xf32>, tensor<2x3x5x128xf32>) -> tensor<2x3x5x128xf32>
  %empty = tensor.empty() : tensor<2x2x2x132xi8>
  %q = hart.launch @quantize(%shared) outs(%empty) <thread = 2> : (tensor<2x3x5x128xf32>, tensor<2x2x2x132xi8>) -> tensor<2x2x2x132xi8>
  %result_empty = tensor.empty() : tensor<2x2x2x132xi8>
  %result = hart.transfer %q to(%result_empty) <from = 2> : (tensor<2x2x2x132xi8>, tensor<2x2x2x132xi8>) -> tensor<2x2x2x132xi8>
  bufferization.materialize_in_destination %result in restrict writable %out : (tensor<2x2x2x132xi8>,memref<2x2x2x132xi8>) -> ()
  return
}
func.func private @quantize(%input: tensor<2x3x5x128xf32> {bufferization.buffer_layout = affine_map<(a,b,c,d)->(a,b,c,d)>, bufferization.writable = false}, %out: tensor<2x2x2x132xi8> {bufferization.buffer_layout = affine_map<(a,b,c,d)->(a,b,c,d)>, bufferization.writable = true}) -> tensor<2x2x2x132xi8> attributes {bbacc.target = "ffn", hart.program_bank = 40 : i64} {
  %slice = tensor.extract_slice %input[0,0,1,0] [2,2,2,128] [1,1,2,1] : tensor<2x3x5x128xf32> to tensor<2x2x2x128xf32>
  %q = "frontend.fp32_to_mxint8"(%slice) {axis = 3 : i64} : (tensor<2x2x2x128xf32>) -> tensor<2x2x2x132xi8>
  %written = bufferization.materialize_in_destination %q in %out : (tensor<2x2x2x132xi8>, tensor<2x2x2x132xi8>) -> tensor<2x2x2x132xi8>
  return %written : tensor<2x2x2x132xi8>
}
