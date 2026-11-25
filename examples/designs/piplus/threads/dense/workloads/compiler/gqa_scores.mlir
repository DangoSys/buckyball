// The scores of grouped-query attention, as the prefix computes them: three
// query heads a key head, the A rows of a key head (query head, token) two
// strided axes of the queries, quantized by DMA without a CPU gather; N = 5
// keys, whose last columns go out by one strided DMA.
func.func @compute(%q: memref<1x5x6x64xf32>, %k: memref<1x5x2x64xf32>, %out: memref<2x15x5xf32>) attributes {llvm.emit_c_interface} {
  %qt = bufferization.to_tensor %q restrict : memref<1x5x6x64xf32> to tensor<1x5x6x64xf32>
  %kt = bufferization.to_tensor %k restrict : memref<1x5x2x64xf32> to tensor<1x5x2x64xf32>
  %0 = "frontend.fp32_to_mxint8"(%qt) <{axis = 3 : i64}> : (tensor<1x5x6x64xf32>) -> tensor<1x5x6x66xi8>
  %1 = "frontend.mxint8_to_fp32"(%0) <{axis = 3 : i64}> : (tensor<1x5x6x66xi8>) -> tensor<1x5x6x64xf32>
  %expanded = tensor.expand_shape %1 [[0], [1], [2, 3], [4]] output_shape [1, 5, 2, 3, 64] : tensor<1x5x6x64xf32> into tensor<1x5x2x3x64xf32>
  %2 = tensor.empty() : tensor<1x2x3x5x64xf32>
  %transposed = linalg.transpose ins(%expanded : tensor<1x5x2x3x64xf32>) outs(%2 : tensor<1x2x3x5x64xf32>) permutation = [0, 2, 3, 1, 4]
  %collapsed = tensor.collapse_shape %transposed [[0, 1], [2, 3], [4]] : tensor<1x2x3x5x64xf32> into tensor<2x15x64xf32>
  %3 = "frontend.fp32_to_mxint8"(%kt) <{axis = 3 : i64}> : (tensor<1x5x2x64xf32>) -> tensor<1x5x2x66xi8>
  %4 = "frontend.mxint8_to_fp32"(%3) <{axis = 3 : i64}> : (tensor<1x5x2x66xi8>) -> tensor<1x5x2x64xf32>
  %5 = tensor.empty() : tensor<1x2x5x64xf32>
  %transposed_0 = linalg.transpose ins(%4 : tensor<1x5x2x64xf32>) outs(%5 : tensor<1x2x5x64xf32>) permutation = [0, 2, 1, 3]
  %6 = tensor.empty() : tensor<1x2x64x5xf32>
  %transposed_1 = linalg.transpose ins(%transposed_0 : tensor<1x2x5x64xf32>) outs(%6 : tensor<1x2x64x5xf32>) permutation = [0, 1, 3, 2]
  %collapsed_2 = tensor.collapse_shape %transposed_1 [[0, 1], [2], [3]] : tensor<1x2x64x5xf32> into tensor<2x64x5xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %7 = tensor.empty() : tensor<2x15x5xf32>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<2x15x5xf32>) -> tensor<2x15x5xf32>
  %9 = linalg.batch_matmul ins(%collapsed, %collapsed_2 : tensor<2x15x64xf32>, tensor<2x64x5xf32>) outs(%8 : tensor<2x15x5xf32>) -> tensor<2x15x5xf32>
  bufferization.materialize_in_destination %9 in writable %out : (tensor<2x15x5xf32>, memref<2x15x5xf32>) -> ()
  return
}
