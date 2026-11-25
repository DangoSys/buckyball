memref.global "private" constant @weight : memref<35072xi8> = dense<2> alignment = 4096
func.func private @check_result(memref<8x400xi32>)
func.func @main() -> i8 {
  %memory = memref.get_global @weight : memref<35072xi8>
  %tensor = bufferization.to_tensor %memory restrict : memref<35072xi8> to tensor<35072xi8>
  %slice = tensor.extract_slice %tensor[3072] [32000] [1] : tensor<35072xi8> to tensor<32000xi8>
  %weight = tensor.expand_shape %slice [[0, 1]] output_shape [80, 400] : tensor<32000xi8> into tensor<80x400xi8>
  %input = arith.constant dense<1> : tensor<8x80xi8>
  %bias = arith.constant dense<3> : tensor<8x400xi32>
  %result = bbacc.custom "matmul_i8"(%input, %weight, %bias) : (tensor<8x80xi8>, tensor<80x400xi8>, tensor<8x400xi32>) -> tensor<8x400xi32>
  %output = bufferization.to_buffer %result : tensor<8x400xi32> to memref<8x400xi32>
  func.call @check_result(%output) : (memref<8x400xi32>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
