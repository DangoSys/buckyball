func.func private @check_result(memref<1x4x4x16xi8>) -> ()

func.func @main() -> i8 {
  %i8_0 = arith.constant 0 : i8
  %i32_0 = arith.constant 0 : i32
  %f32_1 = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c9 = arith.constant 9 : index
  %c16 = arith.constant 16 : index
  %input = memref.alloc() alignment = 64 : memref<1x16x16x3xi8>
  %weight0 = memref.alloc() alignment = 64 : memref<1x3x16x16xi8>
  %bias0 = memref.alloc() alignment = 64 : memref<16xi32>
  %scale0 = memref.alloc() alignment = 64 : memref<16xf32>
  %lut = memref.alloc() alignment = 64 : memref<1xi8>
  %middle = memref.alloc() alignment = 64 : memref<1x8x8x16xi8>
  %weight1 = memref.alloc() alignment = 64 : memref<3x3x16x1xi8>
  %bias1 = memref.alloc() alignment = 64 : memref<16xi32>
  %scale1 = memref.alloc() alignment = 64 : memref<16xf32>
  %output = memref.alloc() alignment = 64 : memref<1x4x4x16xi8>
  scf.for %y = %c0 to %c16 step %c1 {
    scf.for %x = %c0 to %c16 step %c1 {
      scf.for %ic = %c0 to %c3 step %c1 {
        %a = arith.muli %y, %c3 : index
        %b = arith.muli %x, %c2 : index
        %d = arith.addi %a, %b : index
        %e = arith.addi %d, %ic : index
        %r = arith.remui %e, %c7 : index
        %s = arith.subi %r, %c3 : index
        %v = arith.index_cast %s : index to i8
        memref.store %v, %input[%c0, %y, %x, %ic] : memref<1x16x16x3xi8>
      }
    }
  }
  scf.for %ic = %c0 to %c3 step %c1 {
    scf.for %p = %c0 to %c16 step %c1 {
      scf.for %oc = %c0 to %c16 step %c1 {
        %oc3 = arith.muli %oc, %c3 : index
        %a = arith.addi %p, %ic : index
        %b = arith.addi %a, %oc3 : index
        %r = arith.remui %b, %c5 : index
        %s = arith.subi %r, %c2 : index
        %v = arith.index_cast %s : index to i8
        memref.store %v, %weight0[%c0, %ic, %p, %oc] : memref<1x3x16x16xi8>
      }
    }
  }
  scf.for %oc = %c0 to %c16 step %c1 {
    %s = arith.subi %oc, %c4 : index
    %v = arith.index_cast %s : index to i32
    memref.store %v, %bias0[%oc] : memref<16xi32>
  }
  linalg.fill ins(%f32_1 : f32) outs(%scale0 : memref<16xf32>)
  linalg.fill ins(%i8_0 : i8) outs(%lut : memref<1xi8>)
  linalg.fill ins(%i8_0 : i8) outs(%middle : memref<1x8x8x16xi8>)
  scf.for %ky = %c0 to %c3 step %c1 {
    scf.for %kx = %c0 to %c3 step %c1 {
      scf.for %c = %c0 to %c16 step %c1 {
        %kx2 = arith.muli %kx, %c2 : index
        %a = arith.addi %ky, %kx2 : index
        %b = arith.addi %a, %c : index
        %r = arith.remui %b, %c5 : index
        %s = arith.subi %r, %c2 : index
        %v = arith.index_cast %s : index to i8
        memref.store %v, %weight1[%ky, %kx, %c, %c0] : memref<3x3x16x1xi8>
      }
    }
  }
  linalg.fill ins(%i32_0 : i32) outs(%bias1 : memref<16xi32>)
  linalg.fill ins(%f32_1 : f32) outs(%scale1 : memref<16xf32>)
  linalg.fill ins(%i8_0 : i8) outs(%output : memref<1x4x4x16xi8>)
  tile.mega_kernel %input %output : memref<1x16x16x3xi8> memref<1x4x4x16xi8> {
    tile.mega_conv2d %input %weight0 %bias0 %scale0 %lut %middle
        {activation = 1 : i64, kernel = 3 : i64, outputScale = 1.0 : f32,
         padHigh = 1 : i64, padLow = 1 : i64, stride = 2 : i64}
        : memref<1x16x16x3xi8> memref<1x3x16x16xi8> memref<16xi32>
          memref<16xf32> memref<1xi8> memref<1x8x8x16xi8>
    tile.mega_conv2d_depthwise %middle %weight1 %bias1 %scale1 %lut %output
        {activation = 0 : i64, kernel = 3 : i64, outputScale = 1.0 : f32,
         padHigh = 1 : i64, padLow = 1 : i64, stride = 2 : i64}
        : memref<1x8x8x16xi8> memref<3x3x16x1xi8> memref<16xi32>
          memref<16xf32> memref<1xi8> memref<1x4x4x16xi8>
  }
  func.call @check_result(%output) : (memref<1x4x4x16xi8>) -> ()
  return %i8_0 : i8
}
