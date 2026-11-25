// SHM-LABEL: func.func @compute(
// SHM: call @stream_submit
#id = affine_map<(m,n)->(m,n)>
func.func @compute(%input: memref<4x544xf32>, %out: memref<4x561xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<4x544xf32> to tensor<4x544xf32>
  %empty = tensor.empty() : tensor<4x544xf32>
  %twice = linalg.generic {indexing_maps=[#id,#id], iterator_types=["parallel","parallel"]} ins(%x : tensor<4x544xf32>) outs(%empty : tensor<4x544xf32>) {
  ^bb0(%v: f32, %unused: f32):
    %value = arith.addf %v, %v : f32
    linalg.yield %value : f32
  } -> tensor<4x544xf32>
  %q = "frontend.fp32_to_mxint8"(%twice) {axis = 1 : i64} : (tensor<4x544xf32>) -> tensor<4x561xi8>
  bufferization.materialize_in_destination %q in restrict writable %out : (tensor<4x561xi8>,memref<4x561xi8>) -> ()
  return
}
