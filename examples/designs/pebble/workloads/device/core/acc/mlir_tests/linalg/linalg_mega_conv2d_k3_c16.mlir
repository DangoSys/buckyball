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
  %weight = memref.alloc() : memref<16x16x3x3xi8>
  %bias = memref.alloc() : memref<16xi32>
  linalg.fill ins(%zero_i8 : i8) outs(%input : memref<1x3x3x16xi8>)
  linalg.fill ins(%zero_i8 : i8) outs(%weight : memref<16x16x3x3xi8>)
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
        memref.store %wv, %weight[%out, %channel, %h, %w] : memref<16x16x3x3xi8>
      }
    }
  }
  scf.for %out = %zero to %sixteen step %one {
    %bval = arith.subi %out, %eight : index
    %bv = arith.index_cast %bval : index to i32
    memref.store %bv, %bias[%out] : memref<16xi32>
  }
  %unit = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
  %weight_step = arith.constant dense<1.0> : tensor<16x1x1x1xf32>
  %it = bufferization.to_tensor %input restrict : memref<1x3x3x16xi8> to tensor<1x3x3x16xi8>
  %wt = bufferization.to_tensor %weight restrict : memref<16x16x3x3xi8> to tensor<16x16x3x3xi8>
  %bt = bufferization.to_tensor %bias restrict : memref<16xi32> to tensor<16xi32>
  %input_float = "frontend.int8_to_fp32"(%it, %unit) : (tensor<1x3x3x16xi8>, tensor<1x1x1x1xf32>) -> tensor<1x3x3x16xf32>
  %q = "frontend.fp32_to_int8"(%input_float, %unit) : (tensor<1x3x3x16xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x3x16xi8>
  %x = "frontend.int8_to_fp32"(%q, %unit) : (tensor<1x3x3x16xi8>, tensor<1x1x1x1xf32>) -> tensor<1x3x3x16xf32>
  %wf = "frontend.int8_to_fp32"(%wt, %weight_step) : (tensor<16x16x3x3xi8>, tensor<16x1x1x1xf32>) -> tensor<16x16x3x3xf32>
  %we = tensor.empty() : tensor<16x3x3x16xf32>
  %weights = linalg.transpose ins(%wf: tensor<16x16x3x3xf32>) outs(%we: tensor<16x3x3x16xf32>) permutation = [0,2,3,1]
  %scale = linalg.generic {indexing_maps = [affine_map<(f,c,h,w)->(0,c,h,w)>, affine_map<(f,c,h,w)->(f,c,h,w)>, affine_map<(f,c,h,w)->(f,c,h,w)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%unit, %weight_step: tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) outs(%weight_step: tensor<16x1x1x1xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %r = arith.mulf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<16x1x1x1xf32>
  %bs = tensor.collapse_shape %scale [[0,1,2,3]] : tensor<16x1x1x1xf32> into tensor<16xf32>
  %bf = "frontend.int32_to_fp32"(%bt, %bs) : (tensor<16xi32>, tensor<16xf32>) -> tensor<16xf32>
  %empty = tensor.empty() : tensor<1x1x1x16xf32>
  %initial = linalg.generic {indexing_maps = [affine_map<(n,h,w,c)->(c)>, affine_map<(n,h,w,c)->(n,h,w,c)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%bf: tensor<16xf32>) outs(%empty: tensor<1x1x1x16xf32>) {
  ^bb0(%a: f32, %o: f32): linalg.yield %a : f32
  } -> tensor<1x1x1x16xf32>
  %result = linalg.conv_2d_nhwc_fhwc {strides = dense<1> : tensor<2xi64>, dilations = dense<1> : tensor<2xi64>} ins(%x, %weights: tensor<1x3x3x16xf32>, tensor<16x3x3x16xf32>) outs(%initial: tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
  %oe = tensor.empty() : tensor<1x16x1x1xf32>
  %nchw = linalg.transpose ins(%result: tensor<1x1x1x16xf32>) outs(%oe: tensor<1x16x1x1xf32>) permutation = [0,3,1,2]
  %memory = bufferization.to_buffer %nchw : tensor<1x16x1x1xf32> to memref<1x16x1x1xf32>
  func.call @check_result(%memory) : (memref<1x16x1x1xf32>) -> ()
  return %zero_i8 : i8
}
