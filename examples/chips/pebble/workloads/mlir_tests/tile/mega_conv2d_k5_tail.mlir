func.func private @check_result(memref<1x16x5x5xf32>)
func.func @main() -> i8 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c16 = arith.constant 16 : index
  %c25 = arith.constant 25 : index
  %zi8 = arith.constant 0 : i8
  %zi32 = arith.constant 0 : i32
  %zf32 = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %input = memref.alloc() alignment = 64 : memref<1x5x5x2xi8>
  %weight = memref.alloc() alignment = 64 : memref<1x2x32x16xi8>
  %bias = memref.alloc() alignment = 64 : memref<16xi32>
  %scale = memref.alloc() alignment = 64 : memref<16xf32>
  %lut = memref.alloc() alignment = 64 : memref<1xi8>
  %output = memref.alloc() alignment = 64 : memref<1x16x5x5xf32>
  linalg.fill ins(%zi8 : i8) outs(%weight : memref<1x2x32x16xi8>)
  linalg.fill ins(%zi32 : i32) outs(%bias : memref<16xi32>)
  linalg.fill ins(%one : f32) outs(%scale : memref<16xf32>)
  linalg.fill ins(%zi8 : i8) outs(%lut : memref<1xi8>)
  linalg.fill ins(%zf32 : f32) outs(%output : memref<1x16x5x5xf32>)
  scf.for %y = %c0 to %c5 step %c1 {
    scf.for %x = %c0 to %c5 step %c1 {
      scf.for %c = %c0 to %c2 step %c1 {
        %y3 = arith.muli %y, %c3 : index
        %x2 = arith.muli %x, %c2 : index
        %s0 = arith.addi %y3, %x2 : index
        %s = arith.addi %s0, %c : index
        %r = arith.remui %s, %c7 : index
        %v = arith.subi %r, %c3 : index
        %i = arith.index_cast %v : index to i8
        memref.store %i, %input[%c0, %y, %x, %c] : memref<1x5x5x2xi8>
      }
    }
  }
  scf.for %c = %c0 to %c2 step %c1 {
    scf.for %k = %c0 to %c25 step %c1 {
      scf.for %o = %c0 to %c16 step %c1 {
        %s0 = arith.addi %k, %c : index
        %s = arith.addi %s0, %o : index
        %r = arith.remui %s, %c3 : index
        %v = arith.subi %r, %c1 : index
        %i = arith.index_cast %v : index to i8
        memref.store %i, %weight[%c0, %c, %k, %o] : memref<1x2x32x16xi8>
      }
    }
  }
  tile.mega_kernel %input %output : memref<1x5x5x2xi8> memref<1x16x5x5xf32> {
    tile.mega_conv2d %input %weight %bias %scale %lut %output
        {activation = 0 : i64, kernel = 5 : i64, outputScale = 1.0 : f32,
         padHigh = 2 : i64, padLow = 2 : i64, stride = 1 : i64}
        : memref<1x5x5x2xi8> memref<1x2x32x16xi8> memref<16xi32>
          memref<16xf32> memref<1xi8> memref<1x16x5x5xf32>
  }
  func.call @check_result(%output) : (memref<1x16x5x5xf32>) -> ()
  %ok = arith.constant 0 : i8
  return %ok : i8
}
