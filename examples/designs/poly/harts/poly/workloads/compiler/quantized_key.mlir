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

func.func @static_wide(%input: memref<2x3x512xf32>, %out: memref<2x3x528xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<2x3x512xf32> to tensor<2x3x512xf32>
  %q = "frontend.fp32_to_mxint8"(%x) {axis = 2 : i64} : (tensor<2x3x512xf32>) -> tensor<2x3x528xi8>
  bufferization.materialize_in_destination %q in writable %out : (tensor<2x3x528xi8>, memref<2x3x528xi8>) -> ()
  return
}

func.func @static_narrow(%input: memref<2x3x32xf32>, %out: memref<2x3x33xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<2x3x32xf32> to tensor<2x3x32xf32>
  %q = "frontend.fp32_to_mxint8"(%x) {axis = 2 : i64} : (tensor<2x3x32xf32>) -> tensor<2x3x33xi8>
  bufferization.materialize_in_destination %q in writable %out : (tensor<2x3x33xi8>, memref<2x3x33xi8>) -> ()
  return
}

func.func @static_full(%input: memref<2x1x4096xf32>, %out: memref<2x1x4224xi8>) attributes {llvm.emit_c_interface} {
  %x = bufferization.to_tensor %input restrict : memref<2x1x4096xf32> to tensor<2x1x4096xf32>
  %q = "frontend.fp32_to_mxint8"(%x) {axis = 2 : i64} : (tensor<2x1x4096xf32>) -> tensor<2x1x4224xi8>
  bufferization.materialize_in_destination %q in writable %out : (tensor<2x1x4224xi8>, memref<2x1x4224xi8>) -> ()
  return
}
