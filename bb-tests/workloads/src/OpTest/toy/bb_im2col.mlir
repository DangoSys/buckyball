// Test for bb_im2col: extract 3x3 patches from 4x16 input
// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec:
// Purpose: verify correctness of im2col operation
// 1. Load input matrix (4x16)
// 2. Im2col with 3x3 kernel extracts patches into output matrix
// 3. Print output matrix to verify patch extraction

// Input: 4x16 matrix
memref.global "private" @input : memref<4x16xi8> = dense<[
  [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
  [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
  [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
  [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8

  %arrayIn = memref.get_global @input : memref<4x16xi8>
  // Output for im2col: 2 output rows, 3*3*1=9 patch cols (padded to 16)
  %arrayOut = memref.alloc() : memref<2x16xi8>

  // Im2col parameters
  %kRow = arith.constant 3 : i64
  %kCol = arith.constant 3 : i64
  %inRow = arith.constant 4 : i64
  %inCol = arith.constant 16 : i64
  %startRow = arith.constant 0 : i64
  %startCol = arith.constant 0 : i64

  // Print input
  buckyball.bb_print_memref %arrayIn : memref<4x16xi8>

  // CHECK: im2col
  buckyball.bb_im2col %arrayIn %arrayOut %kRow %kCol %inRow %inCol %startRow %startCol
    : memref<4x16xi8> memref<2x16xi8>

  // Print output
  buckyball.bb_print_memref %arrayOut : memref<2x16xi8>

  memref.dealloc %arrayOut : memref<2x16xi8>
  return %0 : i8
}
