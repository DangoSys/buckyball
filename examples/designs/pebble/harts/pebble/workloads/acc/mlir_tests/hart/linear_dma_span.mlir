memref.global "private" constant @input : memref<37905xi8> = dense<1> alignment = 4096
func.func private @check_result(memref<17x16xi32>)
func.func @main() -> i8 {
  %memory = memref.get_global @input : memref<37905xi8>
  %tensor = bufferization.to_tensor %memory restrict : memref<37905xi8> to tensor<37905xi8>
  %slice = tensor.extract_slice %tensor[3072] [34833] [1] : tensor<37905xi8> to tensor<34833xi8>
  %input = tensor.expand_shape %slice [[0, 1]] output_shape [17, 2049] : tensor<34833xi8> into tensor<17x2049xi8>
  %weight = arith.constant dense<1> : tensor<2049x16xi8>
  %bias = arith.constant dense<0> : tensor<17x16xi32>
  %result = bbacc.custom "matmul_i8"(%input, %weight, %bias) : (tensor<17x2049xi8>, tensor<2049x16xi8>, tensor<17x16xi32>) -> tensor<17x16xi32>
  %output = bufferization.to_buffer %result : tensor<17x16xi32> to memref<17x16xi32>
  func.call @check_result(%output) : (memref<17x16xi32>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
