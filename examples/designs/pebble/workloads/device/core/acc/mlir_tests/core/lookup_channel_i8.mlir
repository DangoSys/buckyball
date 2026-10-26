func.func private @check_result(memref<4x16xi8>)
func.func @main() -> i8 {
  %zero = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %c17 = arith.constant 17 : index
  %c29 = arith.constant 29 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c4096 = arith.constant 4096 : index
  %input = memref.alloc() : memref<4x16xi8>
  %table = memref.alloc() : memref<4096xi8>
  scf.for %row = %c0 to %c4 step %c1 {
    scf.for %lane = %c0 to %c16 step %c1 {
      %base = arith.muli %row, %c16 : index
      %offset = arith.addi %base, %lane : index
      %scaled = arith.muli %offset, %c29 : index
      %shifted = arith.subi %scaled, %c128 : index
      %value = arith.index_cast %shifted : index to i8
      memref.store %value, %input[%row, %lane] : memref<4x16xi8>
    }
  }
  scf.for %offset = %c0 to %c4096 step %c1 {
    %lane = arith.divui %offset, %c256 : index
    %entry = arith.remui %offset, %c256 : index
    %scaled = arith.muli %lane, %c17 : index
    %sum = arith.addi %entry, %scaled : index
    %shifted = arith.subi %sum, %c128 : index
    %value = arith.index_cast %shifted : index to i8
    memref.store %value, %table[%offset] : memref<4096xi8>
  }
  %x = bufferization.to_tensor %input restrict : memref<4x16xi8> to tensor<4x16xi8>
  %lut = bufferization.to_tensor %table restrict : memref<4096xi8> to tensor<4096xi8>
  %result = custom.op "lookup_i8"(%x, %lut) : (tensor<4x16xi8>, tensor<4096xi8>) -> tensor<4x16xi8>
  %memory = bufferization.to_buffer %result : tensor<4x16xi8> to memref<4x16xi8>
  func.call @check_result(%memory) : (memref<4x16xi8>) -> ()
  return %zero : i8
}
