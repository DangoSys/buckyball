// Quantized convolution in the frontend QDQ form: input steps 0.25 with
// multiplier 4, weight step 0.5, initial output 1. The output is
// 1 + 0.125 * sum(xq * wq), so a wrong step fails the check.
func.func private @check_result(memref<1x3x3x4xf32>) -> ()

func.func @main() -> i8 {
  %zero_i8 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c7 = arith.constant 7 : index
  %c9 = arith.constant 9 : index
  %quarter = arith.constant 0.25 : f32
  %input = memref.alloc() : memref<1x5x5x3xf32>
  %weight = memref.alloc() : memref<3x3x3x4xi8>
  scf.for %h = %c0 to %c5 step %c1 {
    scf.for %w = %c0 to %c5 step %c1 {
      scf.for %c = %c0 to %c3 step %c1 {
        %h3 = arith.muli %h, %c3 : index
        %w5 = arith.muli %w, %c5 : index
        %c7c = arith.muli %c, %c7 : index
        %s0 = arith.addi %h3, %w5 : index
        %s = arith.addi %s0, %c7c : index
        %r = arith.remui %s, %c9 : index
        %v = arith.subi %r, %c4 : index
        %vi = arith.index_cast %v : index to i32
        %vf = arith.sitofp %vi : i32 to f32
        %x = arith.mulf %vf, %quarter : f32
        memref.store %x, %input[%c0, %h, %w, %c] : memref<1x5x5x3xf32>
      }
    }
  }
  scf.for %f = %c0 to %c4 step %c1 {
    scf.for %kh = %c0 to %c3 step %c1 {
      scf.for %kw = %c0 to %c3 step %c1 {
        scf.for %c = %c0 to %c3 step %c1 {
          %f2 = arith.muli %f, %c2 : index
          %kh3 = arith.muli %kh, %c3 : index
          %kw5 = arith.muli %kw, %c5 : index
          %s0 = arith.addi %f2, %kh3 : index
          %s1 = arith.addi %s0, %kw5 : index
          %s = arith.addi %s1, %c : index
          %r = arith.remui %s, %c7 : index
          %v = arith.subi %r, %c3 : index
          %q = arith.index_cast %v : index to i8
          memref.store %q, %weight[%kh, %kw, %c, %f] : memref<3x3x3x4xi8>
        }
      }
    }
  }
  %multiplier = arith.constant dense<4.0> : tensor<1x1x1x1xf32>
  %input_step = arith.constant dense<0.25> : tensor<1x1x1x1xf32>
  %weight_step = arith.constant dense<0.5> : tensor<1x1x1x1xf32>
  %init = arith.constant dense<1.0> : tensor<1x3x3x4xf32>
  %it = bufferization.to_tensor %input restrict : memref<1x5x5x3xf32> to tensor<1x5x5x3xf32>
  %wt = bufferization.to_tensor %weight restrict : memref<3x3x3x4xi8> to tensor<3x3x3x4xi8>
  %iq = "frontend.fp32_to_int8"(%it, %multiplier) : (tensor<1x5x5x3xf32>, tensor<1x1x1x1xf32>) -> tensor<1x5x5x3xi8>
  %id = "frontend.int8_to_fp32"(%iq, %input_step) : (tensor<1x5x5x3xi8>, tensor<1x1x1x1xf32>) -> tensor<1x5x5x3xf32>
  %wd = "frontend.int8_to_fp32"(%wt, %weight_step) : (tensor<3x3x3x4xi8>, tensor<1x1x1x1xf32>) -> tensor<3x3x3x4xf32>
  %result = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%id, %wd : tensor<1x5x5x3xf32>, tensor<3x3x3x4xf32>) outs(%init : tensor<1x3x3x4xf32>) -> tensor<1x3x3x4xf32>
  %memory = bufferization.to_buffer %result : tensor<1x3x3x4xf32> to memref<1x3x3x4xf32>
  func.call @check_result(%memory) : (memref<1x3x3x4xf32>) -> ()
  return %zero_i8 : i8
}
