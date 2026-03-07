// Test for bb_quant and bb_dequant: quantize FP32 to INT8 then dequantize back
// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec:
// Purpose: verify quant -> matmul -> dequant pipeline
// 1. Quantize FP32 input to INT8 using scale factor
// 2. Perform matmul on quantized data
// 3. Dequantize result back to FP32

// FP32 input matrix A: 16x16
memref.global "private" @fp_input_a : memref<16x16xf32> = dense<1.0>
// FP32 input matrix B: 16x16
memref.global "private" @fp_input_b : memref<16x16xf32> = dense<2.0>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  %scale = arith.constant 0.1 : f32

  // Load FP32 inputs
  %fpA = memref.get_global @fp_input_a : memref<16x16xf32>
  %fpB = memref.get_global @fp_input_b : memref<16x16xf32>

  // Allocate INT8 buffers for quantized data
  %qA = memref.alloc() : memref<16x16xi8>
  %qB = memref.alloc() : memref<16x16xi8>

  // Quantize A and B
  // CHECK: quant
  buckyball.bb_quant %fpA %qA %scale : memref<16x16xf32> memref<16x16xi8> f32
  buckyball.bb_quant %fpB %qB %scale : memref<16x16xf32> memref<16x16xi8> f32

  // MatMul on quantized data
  %qC = memref.alloc() : memref<16x16xi8>
  buckyball.bb_matmul %qA %qB %qC : memref<16x16xi8> memref<16x16xi8> memref<16x16xi8>

  // Dequantize result
  %fpC = memref.alloc() : memref<16x16xf32>
  // CHECK: dequant
  buckyball.bb_dequant %qC %fpC %scale : memref<16x16xi8> memref<16x16xf32> f32

  // Print dequantized result
  buckyball.bb_print_memref %fpC : memref<16x16xf32>

  memref.dealloc %qA : memref<16x16xi8>
  memref.dealloc %qB : memref<16x16xi8>
  memref.dealloc %qC : memref<16x16xi8>
  memref.dealloc %fpC : memref<16x16xf32>
  return %0 : i8
}
