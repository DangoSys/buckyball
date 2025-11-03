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

// // === Check if current hartid is 5, otherwise exit ===
func.func @main() -> i8 {
//   %hartid = llvm.inline_asm "csrr $0, mhartid", "=r" : () -> i32
//   // buckyball.print_scalar %hartid : i32

//   %target_hart = arith.constant 5 : i32
//   %is_correct_hart = arith.cmpi eq, %hartid, %target_hart : i32
//   cf.cond_br %is_correct_hart, ^continue, ^exit

// ^exit:
//   // If not hart 5, return 0
//   %error = arith.constant -1 : i8
//   return %error : i8

// ^continue:
  buckyball.multicore
  // %hartid = llvm.inline_asm "csrr $0, mhartid", "=r" : () -> i32
  // buckyball.print_scalar %hartid : i32
  // === Main program ===
  %0 = arith.constant 0 : i8
  %spadAddr = arith.constant 1040 : i64
  %arrayA = memref.get_global @input_matrix : memref<4x16xi8>
  %arrayB = memref.alloc() : memref<4x16xi8>
  // Use mvin to move data from memory to scratchpad
  // CHECK: mvin
  buckyball.bb_mvin %arrayA %spadAddr : memref<4x16xi8> i64
  // Use mvout to move data from scratchpad back to output memory
  // CHECK: mvout
  buckyball.bb_mvout %arrayB %spadAddr : memref<4x16xi8> i64
  // Print moved output matrix
  buckyball.print %arrayB : memref<4x16xi8>
  // Release allocated memory
  memref.dealloc %arrayB : memref<4x16xi8>

  // exit
  %exit_code = arith.constant 0 : i32
  func.call @exit(%exit_code) : (i32) -> ()
  llvm.unreachable
}

func.func private @exit(i32) -> ()
