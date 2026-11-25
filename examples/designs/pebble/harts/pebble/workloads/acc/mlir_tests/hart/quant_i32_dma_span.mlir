memref.global "private" constant @storage : memref<8960xi32> = dense<3> alignment = 4096
func.func private @check_result(memref<1x1x128x64xi8>)
func.func @quantize(%input: tensor<1x1x128x64xi32> {bufferization.buffer_layout = strided<[8192, 8192, 64, 1], offset: ?>, bufferization.writable = false}) -> tensor<1x1x128x64xi8> {
  %scale = arith.constant dense<[[[[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]]]> : tensor<1x1x1x64xf32>
  %result = "frontend.int32_to_int8"(%input, %scale) : (tensor<1x1x128x64xi32>, tensor<1x1x1x64xf32>) -> tensor<1x1x128x64xi8>
  return %result : tensor<1x1x128x64xi8>
}
func.func @main() -> i8 {
  %storage = memref.get_global @storage : memref<8960xi32>
  %view = memref.reinterpret_cast %storage to offset: [768], sizes: [1, 1, 128, 64], strides: [8192, 8192, 64, 1] : memref<8960xi32> to memref<1x1x128x64xi32, strided<[8192, 8192, 64, 1], offset: 768>>
  %input = bufferization.to_tensor %view restrict : memref<1x1x128x64xi32, strided<[8192, 8192, 64, 1], offset: 768>> to tensor<1x1x128x64xi32>
  %result = func.call @quantize(%input) : (tensor<1x1x128x64xi32>) -> tensor<1x1x128x64xi8>
  %memory = bufferization.to_buffer %result : tensor<1x1x128x64xi8> to memref<1x1x128x64xi8>
  func.call @check_result(%memory) : (memref<1x1x128x64xi8>) -> ()
  %success = arith.constant 0 : i8
  return %success : i8
}
