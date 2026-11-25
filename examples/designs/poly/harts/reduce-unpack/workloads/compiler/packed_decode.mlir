module {
  func.func @compute32(%input: memref<?x33xi8, strided<[33, 1], offset: ?>>, %output: memref<?x32xf32, strided<[40, 1], offset: ?>>) attributes {llvm.emit_c_interface} {
    %x = bufferization.to_tensor %input restrict : memref<?x33xi8, strided<[33, 1], offset: ?>> to tensor<?x33xi8>
    %y = "frontend.mxint8_to_fp32"(%x) {axis = 1 : i64} : (tensor<?x33xi8>) -> tensor<?x32xf32>
    bufferization.materialize_in_destination %y in writable %output : (tensor<?x32xf32>, memref<?x32xf32, strided<[40, 1], offset: ?>>) -> ()
    return
  }
  func.func @compute544(%input: memref<?x561xi8, strided<[561, 1], offset: ?>>, %output: memref<?x544xf32, strided<[552, 1], offset: ?>>) attributes {llvm.emit_c_interface} {
    %x = bufferization.to_tensor %input restrict : memref<?x561xi8, strided<[561, 1], offset: ?>> to tensor<?x561xi8>
    %y = "frontend.mxint8_to_fp32"(%x) {axis = 1 : i64} : (tensor<?x561xi8>) -> tensor<?x544xf32>
    bufferization.materialize_in_destination %y in writable %output : (tensor<?x544xf32>, memref<?x544xf32, strided<[552, 1], offset: ?>>) -> ()
    return
  }
  func.func @compute4096(%input: memref<?x4224xi8, strided<[4224, 1], offset: ?>>, %output: memref<?x4096xf32, strided<[4104, 1], offset: ?>>) attributes {llvm.emit_c_interface} {
    %x = bufferization.to_tensor %input restrict : memref<?x4224xi8, strided<[4224, 1], offset: ?>> to tensor<?x4224xi8>
    %y = "frontend.mxint8_to_fp32"(%x) {axis = 1 : i64} : (tensor<?x4224xi8>) -> tensor<?x4096xf32>
    bufferization.materialize_in_destination %y in writable %output : (tensor<?x4096xf32>, memref<?x4096xf32, strided<[4104, 1], offset: ?>>) -> ()
    return
  }
}
