func.func private @check_result(memref<16x16xi8>) -> ()

func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : i8
  %c5 = arith.constant 5 : i8
  %c16 = arith.constant 16 : index
  %input = memref.alloc() alignment = 64 : memref<16x16xi8>
  scf.for %i = %c0 to %c16 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %ii = arith.index_cast %i : index to i8
      %jj = arith.index_cast %j : index to i8
      %iv = arith.muli %ii, %c3 : i8
      %jv = arith.muli %jj, %c5 : i8
      %v = arith.addi %iv, %jv : i8
      memref.store %v, %input[%i, %j] : memref<16x16xi8>
    }
  }
  %it = bufferization.to_tensor %input restrict : memref<16x16xi8> to tensor<16x16xi8>
  %empty = tensor.empty() : tensor<16x16xi8>
  %result = linalg.transpose ins(%it: tensor<16x16xi8>) outs(%empty: tensor<16x16xi8>) permutation = [1,0]
  %memory = bufferization.to_buffer %result : tensor<16x16xi8> to memref<16x16xi8>
  func.call @check_result(%memory) : (memref<16x16xi8>) -> ()
  return %zero : i8
}
