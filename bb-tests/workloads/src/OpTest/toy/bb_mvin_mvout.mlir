// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec:
// 目的：mvin和mvout指令的正确性
// 1. 打印输入矩阵
// 2. 打印搬移前目标地址的矩阵  [CHECK] 打印结果应为全0矩阵
// 3. 使用mvin将数据从内存搬到暂存器
// 4. 使用mvout将数据从暂存器搬回输出内存
// 5. 打印搬移后目标地址的矩阵  [CHECK] 打印结果应该与输入矩阵相同

// Matrix B: 4x16 (test matrix with simple pattern)
memref.global "private" @input_matrix : memref<4x16xi8> = dense<[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                                                                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]>

func.func @main() -> i8 {
//   // === 检查当前hartid是否为5，否则退出 ===
//   %hartid = llvm.inline_asm "csrr $0, mhartid", "=r" : () -> i32
//   // buckyball.print_scalar %hartid : i32

//   %target_hart = arith.constant 5 : i32
//   %is_correct_hart = arith.cmpi eq, %hartid, %target_hart : i32
//   cf.cond_br %is_correct_hart, ^continue, ^exit

// ^exit:
//   // 如果不是hart 5，返回 0
//   %error = arith.constant -1 : i8
//   return %error : i8

// ^continue:
  buckyball.multicore
  // %hartid = llvm.inline_asm "csrr $0, mhartid", "=r" : () -> i32
  // buckyball.print_scalar %hartid : i32
  // === 主程序 ===
  %0 = arith.constant 0 : i8
  %spadAddr = arith.constant 1040 : i64
  %arrayA = memref.get_global @input_matrix : memref<4x16xi8>
  %arrayB = memref.alloc() : memref<4x16xi8>
  // 使用mvin将数据从内存搬到暂存器
  // CHECK: mvin
  buckyball.bb_mvin %arrayA %spadAddr : memref<4x16xi8> i64
  // 使用mvout将数据从暂存器搬回输出内存
  // CHECK: mvout
  buckyball.bb_mvout %arrayB %spadAddr : memref<4x16xi8> i64
  // 打印搬移后的输出矩阵
  buckyball.print %arrayB : memref<4x16xi8>
  // 释放分配的内存
  memref.dealloc %arrayB : memref<4x16xi8>

  // exit
  %exit_code = arith.constant 0 : i32
  func.call @exit(%exit_code) : (i32) -> ()
  llvm.unreachable
}

func.func private @exit(i32) -> ()
