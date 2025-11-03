// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec:
// Purpose: verify correctness of DMA module's long read/write operations
// 1. Dynamically generate large input matrix, repeatedly fill with data 0~127
// 2. Print matrix at target address before move [CHECK1] print result should be all-zero matrix
// 3. Use mvin to read large amount of data from memory into scratchpad
// 4. Use mvout to write large amount of data from scratchpad to memory
// 5. Print matrix at target address after move [CHECK2] print result should be same as input matrix

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c127 = arith.constant 127 : index
  // Scratchpad start address
  %spadAddr = arith.constant 0 : i64

  // ========== Row count configuration area (only modify here) ==========
  // Total number of rows
  %total_rows = arith.constant 1024 : index
  // Last two rows start = total_rows - 2
  %last_rows_start = arith.constant 1022 : index
  // %offset_elements = arith.constant 16336 : index  // Offset = last_rows_start × 16
  // Offset = last_rows_start × 16
  %offset_elements = arith.constant 16352 : index
  // ================================================

  // Allocate large matrix
  %arrayA = memref.alloc() {alignment = 16} : memref<1023x16xi8>
  %arrayB = memref.alloc() {alignment = 16} : memref<1023x16xi8>

  // Use linalg.fill to initialize arrayB to 0
  linalg.fill ins(%0 : i8) outs(%arrayB : memref<1023x16xi8>)

  // Dynamically fill arrayA: repeatedly fill with 0~127
  scf.for %i = %c0 to %total_rows step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %row_offset = arith.muli %i, %c16 : index
      %linear_idx = arith.addi %row_offset, %j : index
      %mod_val = arith.remui %linear_idx, %c127 : index
      %val = arith.index_cast %mod_val : index to i8
      memref.store %val, %arrayA[%i, %j] : memref<1023x16xi8>
    }
  }

  // Print first two rows and last two rows of input matrix
  %arrayA_head = memref.subview %arrayA[0, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1]>>
  %arrayA_tail = memref.subview %arrayA[1021, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1], offset: 16336>>
  buckyball.print %arrayA_head : memref<2x16xi8, strided<[16, 1]>>
  buckyball.print %arrayA_tail : memref<2x16xi8, strided<[16, 1], offset: 16336>>

  // Print first two rows and last two rows of target matrix before move [CHECK1]
  %arrayB_head = memref.subview %arrayB[0, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1]>>
  %arrayB_tail = memref.subview %arrayB[1021, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1], offset: 16336>>
  buckyball.print %arrayB_head : memref<2x16xi8, strided<[16, 1]>>
  buckyball.print %arrayB_tail : memref<2x16xi8, strided<[16, 1], offset: 16336>>

  // Execute long read/write operations
  // Step 1: long read - read large amount of data from memory into scratchpad
  // CHECK: mvin
  buckyball.bb_mvin %arrayA %spadAddr : memref<1023x16xi8> i64

  // Step 2: long write - write large amount of data from scratchpad to memory
  // CHECK: mvout
  buckyball.bb_mvout %arrayB %spadAddr : memref<1023x16xi8> i64

  // Print first two rows and last two rows of output matrix after move [CHECK2]
  %arrayB_head_after = memref.subview %arrayB[0, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1]>>
  %arrayB_tail_after = memref.subview %arrayB[1021, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1], offset: 16336>>
  buckyball.print %arrayB_head_after : memref<2x16xi8, strided<[16, 1]>>
  buckyball.print %arrayB_tail_after : memref<2x16xi8, strided<[16, 1], offset: 16336>>

  // Release allocated memory
  memref.dealloc %arrayA : memref<1023x16xi8>
  memref.dealloc %arrayB : memref<1023x16xi8>
  return %0 : i8
}
