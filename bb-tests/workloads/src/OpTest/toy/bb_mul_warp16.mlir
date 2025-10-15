// Test for bb_mul_warp16: 16x16 * 16x16 = 16x16 matrix multiplication
// RUN: %run

// Matrix A: 16x16 (identity-like matrix for easier verification)
memref.global "private" @matrix_a : memref<16x16xi8> = dense<[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [-1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                              [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                              [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                              [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                              [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]>

// Matrix B: 16x16 (test matrix with simple pattern)
memref.global "private" @matrix_b : memref<16x16xi8> = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]>

memref.global "private" @val : memref<1x1xi8> = dense<[[1]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8

  // Scratchpad addresses
  %spadAddr_a = arith.constant 0: i64    // Matrix A (16x16) at spad address 0-15
  %spadAddr_b = arith.constant 4096 : i64   // Matrix B (16x16) at spad address 16-31
  %spadAddr_c = arith.constant 8192 : i64   // Result C (16x16) at spad address 32-47
  // Get global matrices
  %arrayA = memref.get_global @matrix_a : memref<16x16xi8>
  %arrayB = memref.get_global @matrix_b : memref<16x16xi8>
  %arrayC = memref.alloc() : memref<16x16xi8>
  %val = memref.get_global @val : memref<1x1xi8>
  // Initialize result matrix C to zero

  // MVIN: Move matrices to scratchpad
  buckyball.bb_mvin %arrayA %spadAddr_a : memref<16x16xi8> i64
  buckyball.bb_mvin %arrayB %spadAddr_b : memref<16x16xi8> i64
  buckyball.print %val : memref<1x1xi8>
  // WARP16: Matrix multiplication (16x16 * 16x16)
  // aSpAddr = 0 (Matrix A address)
  // bSpAddr = 20 (Matrix B address)
  // cSpAddr = 40 (Result C address)
  // nLen = 16 (matrix dimension)
  %nLen = arith.constant 16 : i64
  buckyball.bb_mul_warp16 %spadAddr_a %spadAddr_b %spadAddr_c %nLen : i64 i64 i64 i64
  // MVOUT: Move result from scratchpad to memory
  buckyball.print %val : memref<1x1xi8>
  buckyball.bb_mvout %arrayC %spadAddr_c : memref<16x16xi8> i64

  // Print result matrix
  // Expected: Since A is identity matrix, result should be same as matrix B
  buckyball.print %arrayC : memref<16x16xi8>

  memref.dealloc %arrayC : memref<16x16xi8>
  return %0 : i8
}
