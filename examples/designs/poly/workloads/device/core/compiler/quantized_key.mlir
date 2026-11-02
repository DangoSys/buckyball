func.func @compute(%input: memref<2x64x129xf32>, %out: memref<2x66x129xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<2x64x129xf32> to tensor<2x64x129xf32>
  %q = "frontend.fp32_to_mxint8"(%x) {axis = 1 : i64} : (tensor<2x64x129xf32>) -> tensor<2x66x129xi8>
  bufferization.materialize_in_destination %q in writable %out : (tensor<2x66x129xi8>, memref<2x66x129xi8>) -> ()
  return
}

func.func @dynamic_inner(%input: memref<2x?x129xf32>, %out: memref<2x?x129xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<2x?x129xf32> to tensor<2x?x129xf32>
  %q = "frontend.fp32_to_mxint8"(%x) {axis = 1 : i64} : (tensor<2x?x129xf32>) -> tensor<2x?x129xi8>
  bufferization.materialize_in_destination %q in writable %out : (tensor<2x?x129xi8>, memref<2x?x129xi8>) -> ()
  return
}

func.func @dynamic_last(%input: memref<2x129x?xf32>, %out: memref<2x129x?xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<2x129x?xf32> to tensor<2x129x?xf32>
  %q = "frontend.fp32_to_mxint8"(%x) {axis = 2 : i64} : (tensor<2x129x?xf32>) -> tensor<2x129x?xi8>
  bufferization.materialize_in_destination %q in writable %out : (tensor<2x129x?xi8>, memref<2x129x?xi8>) -> ()
  return
}

func.func @static_last(%input: memref<2x129x64xf32>, %out: memref<2x129x66xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<2x129x64xf32> to tensor<2x129x64xf32>
  %q = "frontend.fp32_to_mxint8"(%x) {axis = 2 : i64} : (tensor<2x129x64xf32>) -> tensor<2x129x66xi8>
  bufferization.materialize_in_destination %q in writable %out : (tensor<2x129x66xi8>, memref<2x129x66xi8>) -> ()
  return
}
