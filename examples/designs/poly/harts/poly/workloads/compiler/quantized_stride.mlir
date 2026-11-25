func.func @static_stride(%input: memref<2x32x1025xf32>, %out: memref<2x33x1025xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<2x32x1025xf32> to tensor<2x32x1025xf32>
  %q = "frontend.fp32_to_mxint8"(%x) {axis = 1 : i64} : (tensor<2x32x1025xf32>) -> tensor<2x33x1025xi8>
  bufferization.materialize_in_destination %q in writable %out : (tensor<2x33x1025xi8>, memref<2x33x1025xi8>) -> ()
  return
}
