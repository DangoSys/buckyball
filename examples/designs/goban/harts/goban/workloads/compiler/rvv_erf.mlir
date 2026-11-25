func.func @compute(%a: memref<2x36xf32>, %out: memref<2x36xf32>) attributes {llvm.emit_c_interface} {
  %at = bufferization.to_tensor %a restrict : memref<2x36xf32> to tensor<2x36xf32>
  %y = math.erf %at : tensor<2x36xf32>
  bufferization.materialize_in_destination %y in writable %out : (tensor<2x36xf32>, memref<2x36xf32>) -> ()
  return
}
