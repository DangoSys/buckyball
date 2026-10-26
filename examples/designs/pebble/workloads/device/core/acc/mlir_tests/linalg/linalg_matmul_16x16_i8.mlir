func.func private @check_result(memref<16x16xi32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_i32 = arith.constant 0 : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %seven = arith.constant 7 : index
  %dim = arith.constant 16 : index
  %a = memref.alloc() alignment = 64 : memref<16x16xi8>
  %b = memref.alloc() alignment = 64 : memref<16x16xi8>

  scf.for %row = %zero to %dim step %one {
    scf.for %column = %zero to %dim step %one {
      %r3 = arith.muli %row, %three : index
      %c5 = arith.muli %column, %five : index
      %asum = arith.addi %r3, %c5 : index
      %arem = arith.remui %asum, %seven : index
      %aval = arith.subi %arem, %three : index
      %av = arith.index_cast %aval : index to i8
      %r2 = arith.muli %row, %two : index
      %c3 = arith.muli %column, %three : index
      %bsum = arith.addi %r2, %c3 : index
      %brem = arith.remui %bsum, %five : index
      %bval = arith.subi %brem, %two : index
      %bv = arith.index_cast %bval : index to i8
      memref.store %av, %a[%row, %column] : memref<16x16xi8>
      memref.store %bv, %b[%row, %column] : memref<16x16xi8>
    }
  }
  %at = bufferization.to_tensor %a restrict : memref<16x16xi8> to tensor<16x16xi8>
  %bt = bufferization.to_tensor %b restrict : memref<16x16xi8> to tensor<16x16xi8>
  %init = arith.constant dense<0> : tensor<16x16xi32>
  %result = linalg.matmul ins(%at, %bt: tensor<16x16xi8>, tensor<16x16xi8>) outs(%init: tensor<16x16xi32>) -> tensor<16x16xi32>
  %memory = bufferization.to_buffer %result : tensor<16x16xi32> to memref<16x16xi32>
  func.call @check_result(%memory) : (memref<16x16xi32>) -> ()
  return %zero_i8 : i8
}
