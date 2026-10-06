func.func private @check_result(memref<17x17xf32>) -> ()
func.func @main() -> i8 {
  %z8 = arith.constant 0 : i8
  %zf = arith.constant 0.0 : f32
  %half = arith.constant 0.5 : f32
  %z = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %seven = arith.constant 7 : index
  %eight = arith.constant 8 : index
  %dim = arith.constant 17 : index
  %inner = arith.constant 65 : index
  %input = memref.alloc() alignment = 64 : memref<17x65xi8>
  %weight = memref.alloc() alignment = 64 : memref<65x17xi8>
  %bias = memref.alloc() alignment = 64 : memref<17xi32>
  %scale = memref.alloc() alignment = 64 : memref<17xf32>
  %lut = memref.alloc() alignment = 64 : memref<1xi8>
  %output = memref.alloc() alignment = 64 : memref<17x17xf32>
  scf.for %r = %z to %dim step %one {
    scf.for %k = %z to %inner step %one {
      %sum = arith.addi %r, %k : index
      %rem = arith.remui %sum, %seven : index
      %v = arith.subi %rem, %three : index
      %i8 = arith.index_cast %v : index to i8
      memref.store %i8, %input[%r, %k] : memref<17x65xi8>
    }
    %b = arith.subi %r, %eight : index
    %b32 = arith.index_cast %b : index to i32
    memref.store %b32, %bias[%r] : memref<17xi32>
  }
  scf.for %k = %z to %inner step %one {
    scf.for %c = %z to %dim step %one {
      %k2 = arith.muli %k, %two : index
      %c3 = arith.muli %c, %three : index
      %sum = arith.addi %k2, %c3 : index
      %rem = arith.remui %sum, %five : index
      %v = arith.subi %rem, %two : index
      %i8 = arith.index_cast %v : index to i8
      memref.store %i8, %weight[%k, %c] : memref<65x17xi8>
    }
  }
  linalg.fill ins(%half : f32) outs(%scale : memref<17xf32>)
  linalg.fill ins(%z8 : i8) outs(%lut : memref<1xi8>)
  linalg.fill ins(%zf : f32) outs(%output : memref<17x17xf32>)
  tile.mega_kernel %input %output : memref<17x65xi8> memref<17x17xf32> {
    tile.mega_matmul %input %weight %bias %scale %lut %output
      {activation = 0 : i64, outputScale = 1.0 : f32}
      : memref<17x65xi8> memref<65x17xi8> memref<17xi32>
        memref<17xf32> memref<1xi8> memref<17x17xf32>
  }
  func.call @check_result(%output) : (memref<17x17xf32>) -> ()
  memref.dealloc %input : memref<17x65xi8>
  memref.dealloc %weight : memref<65x17xi8>
  memref.dealloc %bias : memref<17xi32>
  memref.dealloc %scale : memref<17xf32>
  memref.dealloc %lut : memref<1xi8>
  memref.dealloc %output : memref<17x17xf32>
  return %z8 : i8
}
