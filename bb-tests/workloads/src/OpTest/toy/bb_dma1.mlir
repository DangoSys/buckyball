// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec:
// Purpose: verify correctness of DMA module when facing 16-byte aligned addresses
// 1. Print input matrix
// 2. Print matrix at target address before move [CHECK1] print result should be all-zero matrix
// 3. Use mvin to move data from 16-byte aligned source address to scratchpad
// 4. Use mvout to move data from scratchpad to 16-byte aligned target address
// 5. Print matrix at target address after move [CHECK2] print result should be same as input matrix

memref.global "private" @input_matrix_aligned : memref<4x16xi8> = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                                         [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                                                                         [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                                                                         [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  // 16-byte aligned scratchpad address
  %spadAddr16 = arith.constant 16 : i64
  // 32-byte aligned scratchpad address
  %spadAddr32 = arith.constant 32 : i64

  %arrayA = memref.get_global @input_matrix_aligned : memref<4x16xi8>
  %arrayB = memref.alloc() {alignment = 16} : memref<4x16xi8>

  // Print input matrix
  // Print target matrix before move [CHECK1]
  buckyball.print %arrayA : memref<4x16xi8>
  buckyball.print %arrayB : memref<4x16xi8>

  // Use mvin to move data from 16-byte aligned address to scratchpad
  // CHECK: mvin
  // Use mvout to move data from scratchpad to 16-byte aligned target address
  buckyball.bb_mvin %arrayA %spadAddr16 : memref<4x16xi8> i64
  // CHECK: mvout
  buckyball.bb_mvout %arrayB %spadAddr16 : memref<4x16xi8> i64

  // Print output matrix after move [CHECK2]
  buckyball.print %arrayB : memref<4x16xi8>

  // Release allocated memory
  memref.dealloc %arrayB : memref<4x16xi8>
  return %0 : i8
}
