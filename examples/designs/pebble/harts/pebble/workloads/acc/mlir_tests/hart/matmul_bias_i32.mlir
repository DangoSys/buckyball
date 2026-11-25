#id = affine_map<(m, n) -> (m, n)>
func.func private @check_result(memref<1x16xi32>)
func.func @main() -> i8 {
  %success = arith.constant 0 : i8
  %input = arith.constant dense<1> : tensor<1x16xi8>
  %weight = arith.constant dense<1> : tensor<16x16xi8>
  %zero = arith.constant dense<0> : tensor<1x16xi32>
  %bias = arith.constant dense<[[2147483647, -2147483648, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2]]> : tensor<1x16xi32>
  %raw = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%input, %weight : tensor<1x16xi8>, tensor<16x16xi8>) outs(%zero : tensor<1x16xi32>) -> tensor<1x16xi32>
  %observed = frontend.trace_end %raw {cycle = false, id = 0 : i64, id_path = [0], tag = "matmul", tensor = false} : tensor<1x16xi32>
  %empty = tensor.empty() : tensor<1x16xi32>
  %result = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%observed, %bias : tensor<1x16xi32>, tensor<1x16xi32>) outs(%empty : tensor<1x16xi32>) {
  ^bb0(%a: i32, %b: i32, %c: i32):
    %sum = arith.addi %a, %b : i32
    linalg.yield %sum : i32
  } -> tensor<1x16xi32>
  %memory = bufferization.to_buffer %result : tensor<1x16xi32> to memref<1x16xi32>
  func.call @check_result(%memory) : (memref<1x16xi32>) -> ()
  return %success : i8
}
