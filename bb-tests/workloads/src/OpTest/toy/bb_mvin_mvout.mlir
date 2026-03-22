// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec:
// Purpose: correctness of mvin and mvout instructions
// 1. Print input matrix
// 2. Print matrix at target address before move [CHECK] print result should be all-zero matrix
// 3. Use mvin to move data from memory to scratchpad
// 4. Use mvout to move data from scratchpad back to output memory
// 5. Print matrix at target address after move [CHECK] print result should be same as input matrix

// Matrix B: 4x16 (test matrix with simple pattern)
memref.global "private" @input_matrix : memref<4x16xi8> = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]>

func.func @main() -> i8 {
  // === Main program ===
  %0 = arith.constant 0 : i8
  %bankId = arith.constant 0 : i64
  %depth = arith.constant 4 : i64
  %stride = arith.constant 1 : i64
  %arrayA = memref.get_global @input_matrix : memref<4x16xi8>
  %arrayB = memref.alloc() : memref<4x16xi8>
  // Allocate bank 0 before mvin
  buckyball.bb_mset %bankId : i64
  // Use mvin to move data from memory to bank 0
  // CHECK: mvin
  buckyball.bb_mvin %arrayA %bankId %depth %stride : memref<4x16xi8> i64 i64 i64 i64
  // Use mvout to move data from bank 0 back to output memory
  // CHECK: mvout
  buckyball.bb_mvout %arrayB %bankId %depth %stride : memref<4x16xi8> i64 i64 i64 i64
  // Print moved output matrix
  buckyball.bb_print_memref %arrayB : memref<4x16xi8>
  // Release allocated memory
  memref.dealloc %arrayB : memref<4x16xi8>
  return %0 : i8
}
