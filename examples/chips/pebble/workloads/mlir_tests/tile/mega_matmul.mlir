func.func private @check_result(memref<1x16xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %one_i8 = arith.constant 1 : i8
  %zero_i32 = arith.constant 0 : i32
  %one_f32 = arith.constant 1.0 : f32
  %zero_f32 = arith.constant 0.0 : f32
  %input = memref.alloc() alignment = 64 : memref<1x16xi8>
  %weight = memref.alloc() alignment = 64 : memref<16x16xi8>
  %bias = memref.alloc() alignment = 64 : memref<16xi32>
  %scale = memref.alloc() alignment = 64 : memref<16xf32>
  %lut = memref.alloc() alignment = 64 : memref<1xi8>
  %output = memref.alloc() alignment = 64 : memref<1x16xf32>
  linalg.fill ins(%one_i8 : i8) outs(%input : memref<1x16xi8>)
  linalg.fill ins(%one_i8 : i8) outs(%weight : memref<16x16xi8>)
  linalg.fill ins(%zero_i32 : i32) outs(%bias : memref<16xi32>)
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
