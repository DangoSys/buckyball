// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec:
// Purpose: verify correctness of DMA module's fast alternating read/write
// 1. Print input matrices A and B
// 2. Print matrix at target address before move [CHECK1] print result should be all-zero matrix
// 3. Execute fast alternating operations: use mvin to read row by row, mvout to write
// 4. Print matrix at target address after move [CHECK2] print result should show A and B contents swapped

memref.global "private" @input_matrix_a : memref<2x16xi8> = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                                   [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]>

memref.global "private" @input_matrix_b : memref<2x16xi8> = dense<[[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
                                                                                    [116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 126, 125, 124, 123]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  // Scratchpad address 1
  %spadAddr1 = arith.constant 10 : i64
  // Scratchpad address 2
  %spadAddr2 = arith.constant 50 : i64

  %arrayA = memref.get_global @input_matrix_a : memref<2x16xi8>
  %arrayB = memref.get_global @input_matrix_b : memref<2x16xi8>
  %arrayTemp = memref.alloc() {alignment = 16} : memref<2x16xi8>

  // Print input matrices A and B
  buckyball.print %arrayA : memref<2x16xi8>
  buckyball.print %arrayB : memref<2x16xi8>
  // Print temporary matrix before move [CHECK1]
  buckyball.print %arrayTemp : memref<2x16xi8>

  // Fast alternating mvin/mvout operation sequence
  // Step 1: A -> scratchpad 1
  // CHECK: mvin
  // Step 2: scratchpad 1 -> temp
  buckyball.bb_mvin %arrayA %spadAddr1 : memref<2x16xi8> i64
  // CHECK: mvout
  buckyball.bb_mvout %arrayTemp %spadAddr1 : memref<2x16xi8> i64

  // Step 3: B -> scratchpad 2
  // CHECK: mvin
  // Step 4: scratchpad 2 -> A
  buckyball.bb_mvin %arrayB %spadAddr2 : memref<2x16xi8> i64
  // CHECK: mvout
  buckyball.bb_mvout %arrayA %spadAddr2 : memref<2x16xi8> i64

  // Step 5: temp -> scratchpad 1
  // CHECK: mvin
  // Step 6: scratchpad 1 -> B
  buckyball.bb_mvin %arrayTemp %spadAddr1 : memref<2x16xi8> i64
  // CHECK: mvout
  buckyball.bb_mvout %arrayB %spadAddr1 : memref<2x16xi8> i64

  // Print swapped matrices [CHECK2]
  buckyball.print %arrayA : memref<2x16xi8>
  buckyball.print %arrayB : memref<2x16xi8>

  // Release allocated memory
  memref.dealloc %arrayTemp : memref<2x16xi8>
  return %0 : i8
}
