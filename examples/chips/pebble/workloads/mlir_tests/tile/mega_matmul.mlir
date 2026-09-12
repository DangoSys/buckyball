func.func private @check_result(memref<1x16xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_i32 = arith.constant 0 : i32
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %seven = arith.constant 7 : index
  %eight = arith.constant 8 : index
  %dim = arith.constant 16 : index
  %input = memref.alloc() alignment = 64 : memref<1x16xi8>
  %weight = memref.alloc() alignment = 64 : memref<16x16xi8>
  %bias = memref.alloc() alignment = 64 : memref<16xi32>
  %scale = memref.alloc() alignment = 64 : memref<16xf32>
  %lut = memref.alloc() alignment = 64 : memref<1xi8>
  %output = memref.alloc() alignment = 64 : memref<1x16xf32>
  scf.for %k = %zero to %dim step %one {
    %irem = arith.remui %k, %seven : index
    %ival = arith.subi %irem, %three : index
    %iv = arith.index_cast %ival : index to i8
    memref.store %iv, %input[%zero, %k] : memref<1x16xi8>
    %biasv = arith.subi %k, %eight : index
    %bias32 = arith.index_cast %biasv : index to i32
    memref.store %bias32, %bias[%k] : memref<16xi32>
    scf.for %column = %zero to %dim step %one {
      %k2 = arith.muli %k, %two : index
      %c3 = arith.muli %column, %three : index
      %sum = arith.addi %k2, %c3 : index
      %rem = arith.remui %sum, %five : index
      %value = arith.subi %rem, %two : index
      %wv = arith.index_cast %value : index to i8
      memref.store %wv, %weight[%k, %column] : memref<16x16xi8>
    }
  }
  linalg.fill ins(%one_f32 : f32) outs(%scale : memref<16xf32>)
  linalg.fill ins(%zero_i8 : i8) outs(%lut : memref<1xi8>)
  linalg.fill ins(%zero_f32 : f32) outs(%output : memref<1x16xf32>)
  tile.mega_kernel %input %output : memref<1x16xi8> memref<1x16xf32> {
    tile.mega_matmul %input %weight %bias %scale %lut %output
        {activation = 0 : i64, outputScale = 1.0 : f32}
        : memref<1x16xi8> memref<16x16xi8> memref<16xi32>
          memref<16xf32> memref<1xi8> memref<1x16xf32>
  }
  func.call @check_result(%output) : (memref<1x16xf32>) -> ()
  memref.dealloc %input : memref<1x16xi8>
  memref.dealloc %weight : memref<16x16xi8>
  memref.dealloc %bias : memref<16xi32>
  memref.dealloc %scale : memref<16xf32>
  memref.dealloc %lut : memref<1xi8>
  memref.dealloc %output : memref<1x16xf32>
  return %zero_i8 : i8
}
