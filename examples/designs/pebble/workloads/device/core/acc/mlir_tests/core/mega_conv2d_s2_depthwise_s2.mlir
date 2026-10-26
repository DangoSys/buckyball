func.func private @check_result(memref<1x4x4x16xi8>) -> ()

func.func @main() -> i8 {
  %i8_0 = arith.constant 0 : i8
  %i32_0 = arith.constant 0 : i32
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
  %weight0 = memref.alloc() alignment = 64 : memref<16x3x3x3xi8>
  %bias0 = memref.alloc() alignment = 64 : memref<16xi32>
  %weight1 = memref.alloc() alignment = 64 : memref<16x1x3x3xi8>
  %bias1 = memref.alloc() alignment = 64 : memref<16xi32>
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
    scf.for %p = %c0 to %c9 step %c1 {
      scf.for %oc = %c0 to %c16 step %c1 {
        %oc3 = arith.muli %oc, %c3 : index
        %a = arith.addi %p, %ic : index
        %b = arith.addi %a, %oc3 : index
        %r = arith.remui %b, %c5 : index
        %s = arith.subi %r, %c2 : index
        %v = arith.index_cast %s : index to i8
        %ky_index = arith.divui %p, %c3 : index
        %kx_index = arith.remui %p, %c3 : index
        memref.store %v, %weight0[%oc, %ic, %ky_index, %kx_index] : memref<16x3x3x3xi8>
      }
    }
  }
  scf.for %oc = %c0 to %c16 step %c1 {
    %s = arith.subi %oc, %c4 : index
    %v = arith.index_cast %s : index to i32
    memref.store %v, %bias0[%oc] : memref<16xi32>
  }
  scf.for %ky = %c0 to %c3 step %c1 {
    scf.for %kx = %c0 to %c3 step %c1 {
      scf.for %c = %c0 to %c16 step %c1 {
        %kx2 = arith.muli %kx, %c2 : index
        %a = arith.addi %ky, %kx2 : index
        %b = arith.addi %a, %c : index
        %r = arith.remui %b, %c5 : index
        %s = arith.subi %r, %c2 : index
        %v = arith.index_cast %s : index to i8
        memref.store %v, %weight1[%c, %c0, %ky, %kx] : memref<16x1x3x3xi8>
      }
    }
  }
  linalg.fill ins(%i32_0 : i32) outs(%bias1 : memref<16xi32>)
  %unit = arith.constant dense<1.0> : tensor<1x1x1x1xf32>
  %weight_step = arith.constant dense<1.0> : tensor<16x1x1x1xf32>
  %it = bufferization.to_tensor %input restrict : memref<1x16x16x3xi8> to tensor<1x16x16x3xi8>
  %input_float = "frontend.int8_to_fp32"(%it, %unit) : (tensor<1x16x16x3xi8>, tensor<1x1x1x1xf32>) -> tensor<1x16x16x3xf32>
  %wt = bufferization.to_tensor %weight0 restrict : memref<16x3x3x3xi8> to tensor<16x3x3x3xi8>
  %bt = bufferization.to_tensor %bias0 restrict : memref<16xi32> to tensor<16xi32>
  %dt = bufferization.to_tensor %weight1 restrict : memref<16x1x3x3xi8> to tensor<16x1x3x3xi8>
  %dbt = bufferization.to_tensor %bias1 restrict : memref<16xi32> to tensor<16xi32>
  %first = custom.op "conv2d_i8"(%input_float, %wt, %bt, %unit, %unit, %weight_step) {strides = array<i64: 2, 2>, dilations = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>} : (tensor<1x16x16x3xf32>, tensor<16x3x3x3xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x8x8x16xf32>
  %zero = arith.constant dense<0.0> : tensor<1x8x8x16xf32>
  %relu = linalg.generic {indexing_maps = [affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>, affine_map<(n,h,w,c)->(n,h,w,c)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%first, %zero: tensor<1x8x8x16xf32>, tensor<1x8x8x16xf32>) outs(%zero: tensor<1x8x8x16xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %r = arith.maximumf %a, %b : f32
    linalg.yield %r : f32
  } -> tensor<1x8x8x16xf32>
  %second = custom.op "depthwise_conv2d_i8"(%relu, %dt, %dbt, %unit, %unit, %weight_step) {strides = array<i64: 2, 2>, dilations = array<i64: 1, 1>, padding = array<i64: 1, 1, 1, 1>} : (tensor<1x8x8x16xf32>, tensor<16x1x3x3xi8>, tensor<16xi32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<16x1x1x1xf32>) -> tensor<1x4x4x16xf32>
  %q = "frontend.fp32_to_int8"(%second, %unit) : (tensor<1x4x4x16xf32>, tensor<1x1x1x1xf32>) -> tensor<1x4x4x16xi8>
  %memory = bufferization.to_buffer %q : tensor<1x4x4x16xi8> to memref<1x4x4x16xi8>
  func.call @check_result(%memory) : (memref<1x4x4x16xi8>) -> ()
  return %i8_0 : i8
}
