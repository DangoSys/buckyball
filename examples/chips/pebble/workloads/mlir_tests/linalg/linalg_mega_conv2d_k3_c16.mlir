#input = affine_map<(n, h, w, c) -> (n, h, w, c)>
#weight = affine_map<(n, h, w, c) -> (n, c, 0, 0)>
#channel = affine_map<(n, h, w, c) -> (c)>
#scalar = affine_map<(n, h, w, c) -> (0)>
#output = affine_map<(n, h, w, c) -> (n, c, 0, 0)>

func.func private @check_result(memref<1x16x1x1xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %zero_i32 = arith.constant 0 : i32
  %zero_f32 = arith.constant 0.0 : f32
  %one_i8 = arith.constant 1 : i8
  %one_f32 = arith.constant 1.0 : f32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %three = arith.constant 3 : index
  %five = arith.constant 5 : index
  %eight = arith.constant 8 : index
  %nine = arith.constant 9 : index
  %sixteen = arith.constant 16 : index
  %input = memref.alloc() : memref<1x3x3x16xi8>
  %weight = memref.alloc() : memref<1x16x16x16xi8>
  %bias = memref.alloc() : memref<16xi32>
  %scale = memref.alloc() : memref<16xf32>
  %lut = memref.alloc() : memref<1xi8>
  %output = memref.alloc() : memref<1x16x1x1xf32>
  linalg.fill ins(%zero_i8 : i8) outs(%input : memref<1x3x3x16xi8>)
  linalg.fill ins(%zero_i8 : i8) outs(%weight : memref<1x16x16x16xi8>)
  linalg.fill ins(%one_f32 : f32) outs(%scale : memref<16xf32>)
  linalg.fill ins(%zero_i8 : i8) outs(%lut : memref<1xi8>)
  linalg.fill ins(%zero_f32 : f32) outs(%output : memref<1x16x1x1xf32>)
  scf.for %position = %zero to %nine step %one {
    %p2 = arith.muli %position, %two : index
    %h = arith.divui %position, %three : index
    %w = arith.remui %position, %three : index
    scf.for %channel = %zero to %sixteen step %one {
      %c3 = arith.muli %channel, %three : index
      %isum = arith.addi %p2, %c3 : index
      %irem = arith.remui %isum, %five : index
      %ival = arith.subi %irem, %two : index
      %iv = arith.index_cast %ival : index to i8
      memref.store %iv, %input[%zero, %h, %w, %channel] : memref<1x3x3x16xi8>
      scf.for %out = %zero to %sixteen step %one {
        %wsum0 = arith.addi %c3, %p2 : index
        %wsum = arith.addi %wsum0, %out : index
        %wrem = arith.remui %wsum, %five : index
        %wval = arith.subi %wrem, %two : index
        %wv = arith.index_cast %wval : index to i8
        memref.store %wv, %weight[%zero, %channel, %position, %out] : memref<1x16x16x16xi8>
      }
    }
  }
  scf.for %out = %zero to %sixteen step %one {
    %bval = arith.subi %out, %eight : index
    %bv = arith.index_cast %bval : index to i32
    memref.store %bv, %bias[%out] : memref<16xi32>
  }
  linalg.generic {
      indexing_maps = [#input, #weight, #channel, #channel, #scalar, #output],
      iterator_types = ["parallel", "reduction", "reduction", "parallel"]}
      ins(%input, %weight, %bias, %scale, %lut
          : memref<1x3x3x16xi8>, memref<1x16x16x16xi8>, memref<16xi32>,
            memref<16xf32>, memref<1xi8>)
      outs(%output : memref<1x16x1x1xf32>)
      attrs = {activation = 0 : i64, buckyball.mega_conv2d = true,
               buckyball.mega_kernel = true, final_output = true,
               kernel = 3 : i64, mega_kernel_id = "conv_k3_c16",
               mega_kernel_size = 1 : i64, mega_kernel_stage = 0 : i64,
               output_scale = 1.0 : f32, pad_high = 0 : i64,
               pad_low = 0 : i64, stride = 1 : i64} {
    ^bb0(%in: i8, %w: i8, %b: i32, %s: f32, %l: i8, %out: f32):
      linalg.yield %out : f32
  }
  func.call @check_result(%output) : (memref<1x16x1x1xf32>) -> ()
  memref.dealloc %input : memref<1x3x3x16xi8>
  memref.dealloc %weight : memref<1x16x16x16xi8>
  memref.dealloc %bias : memref<16xi32>
  memref.dealloc %scale : memref<16xf32>
  memref.dealloc %lut : memref<1xi8>
  memref.dealloc %output : memref<1x16x1x1xf32>
  return %zero_i8 : i8
}
