func.func @compute(%input: memref<19xi32>, %scale: memref<19xf32>, %out: memref<19xf32>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<19xi32> to tensor<19xi32>
  %s = bufferization.to_tensor %scale restrict : memref<19xf32> to tensor<19xf32>
  %y = "frontend.int32_to_fp32"(%x, %s) : (tensor<19xi32>, tensor<19xf32>) -> tensor<19xf32>
  bufferization.materialize_in_destination %y in writable %out : (tensor<19xf32>, memref<19xf32>) -> ()
  return
}
