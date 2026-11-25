func.func private @check_result(memref<4x24xi8>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c24 = arith.constant 24 : index
  %c17 = arith.constant 17 : index
  %c29 = arith.constant 29 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c8192 = arith.constant 8192 : index
  %input = memref.alloc() : memref<4x24xi8>
  %table = memref.alloc() : memref<8192xi8>
  scf.for %row = %c0 to %c4 step %c1 {
    scf.for %lane = %c0 to %c24 step %c1 {
      %base = arith.muli %row, %c24 : index
      %offset = arith.addi %base, %lane : index
      %scaled = arith.muli %offset, %c29 : index
      %shifted = arith.subi %scaled, %c128 : index
      %value = arith.index_cast %shifted : index to i8
      memref.store %value, %input[%row, %lane] : memref<4x24xi8>
    }
  }
  scf.for %offset = %c0 to %c8192 step %c1 {
    %lane = arith.divui %offset, %c256 : index
    %entry = arith.remui %offset, %c256 : index
    %scaled = arith.muli %lane, %c17 : index
    %sum = arith.addi %entry, %scaled : index
    %shifted = arith.subi %sum, %c128 : index
    %value = arith.index_cast %shifted : index to i8
    memref.store %value, %table[%offset] : memref<8192xi8>
  }
  %x = bufferization.to_tensor %input restrict : memref<4x24xi8> to tensor<4x24xi8>
  %lut = bufferization.to_tensor %table restrict : memref<8192xi8> to tensor<8192xi8>
  %result = bbacc.custom "lookup_i8"(%x, %lut) : (tensor<4x24xi8>, tensor<8192xi8>) -> tensor<4x24xi8>
  %memory = bufferization.to_buffer %result : tensor<4x24xi8> to memref<4x24xi8>
  func.call @check_result(%memory) : (memref<4x24xi8>) -> ()
  return %zero : i8
}
