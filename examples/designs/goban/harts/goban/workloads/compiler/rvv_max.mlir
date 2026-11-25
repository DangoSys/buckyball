func.func @compute(%input: memref<8x36xf32>, %output: memref<8xf32>) attributes {llvm.emit_c_interface} {
  %a = bufferization.to_tensor %input restrict : memref<8x36xf32> to tensor<8x36xf32>
  %low = arith.constant -3.40282347E+38 : f32
  %empty = tensor.empty() : tensor<8xf32>
  %init = linalg.fill ins(%low : f32) outs(%empty : tensor<8xf32>) -> tensor<8xf32>
  %result = linalg.reduce ins(%a : tensor<8x36xf32>) outs(%init : tensor<8xf32>) dimensions = [1]
    (%value: f32, %acc: f32) {
      %next = arith.maximumf %value, %acc : f32
      linalg.yield %next : f32
    }
  bufferization.materialize_in_destination %result in writable %output : (tensor<8xf32>, memref<8xf32>) -> ()
  return
}
