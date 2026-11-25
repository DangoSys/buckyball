// CHECK-LABEL: func.func @shape_only
// CHECK-NOT: memref.copy %arg0,
// CHECK: return
// CHECK-LABEL: func.func @data_read
// CHECK: memref.copy %arg0,
// CHECK: return
module {
  func.func private @shape(%input: tensor<?xf32>, %out: tensor<1xf32>) -> tensor<1xf32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %input, %c0 : tensor<?xf32>
    %i = arith.index_cast %n : index to i64
    %f = arith.sitofp %i : i64 to f32
    %result = tensor.insert %f into %out[%c0] : tensor<1xf32>
    return %result : tensor<1xf32>
  }
  func.func private @data(%input: tensor<?xf32>, %out: tensor<1xf32>) -> tensor<1xf32> {
    %c0 = arith.constant 0 : index
    %f = tensor.extract %input[%c0] : tensor<?xf32>
    %result = tensor.insert %f into %out[%c0] : tensor<1xf32>
    return %result : tensor<1xf32>
  }
  func.func @shape_only(%old: tensor<?xf32> {bufferization.writable = true}) -> (tensor<?xf32>, tensor<1xf32>) {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %new = tensor.insert %one into %old[%c0] : tensor<?xf32>
    %n = tensor.dim %old, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %input = hart.transfer %old to(%empty) <to = 0> : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %out = tensor.empty() : tensor<1xf32>
    %result = hart.launch @shape(%input) outs(%out) <thread = 0> : (tensor<?xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %new, %result : tensor<?xf32>, tensor<1xf32>
  }
  func.func @data_read(%old: tensor<?xf32> {bufferization.writable = true}) -> (tensor<?xf32>, tensor<1xf32>) {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1.0 : f32
    %new = tensor.insert %one into %old[%c0] : tensor<?xf32>
    %n = tensor.dim %old, %c0 : tensor<?xf32>
    %empty = tensor.empty(%n) : tensor<?xf32>
    %input = hart.transfer %old to(%empty) <to = 0> : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %out = tensor.empty() : tensor<1xf32>
    %result = hart.launch @data(%input) outs(%out) <thread = 0> : (tensor<?xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %new, %result : tensor<?xf32>, tensor<1xf32>
  }
}
