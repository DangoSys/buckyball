// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec: 
// 目的：验证dma模块面对16字节对齐地址时的正确性
// 1. 打印输入矩阵 
// 2. 打印搬移前目标地址的矩阵  [CHECK1] 打印结果应为全0矩阵
// 3. 使用mvin将数据从16字节对齐的源地址搬到暂存器
// 4. 使用mvout将数据从暂存器搬到16字节对齐的目标地址
// 5. 打印搬移后目标地址的矩阵  [CHECK2] 打印结果应该与输入矩阵相同

memref.global "private" @input_matrix_aligned : memref<4x16xi8> = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                                         [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
                                                                         [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                                                                         [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  %spadAddr16 = arith.constant 16 : i64   // 16字节对齐暂存器地址
  %spadAddr32 = arith.constant 32 : i64   // 32字节对齐暂存器地址
  
  %arrayA = memref.get_global @input_matrix_aligned : memref<4x16xi8>
  %arrayB = memref.alloc() {alignment = 16} : memref<4x16xi8>
  
  // 打印输入矩阵
  buckyball.print %arrayA : memref<4x16xi8>
  // 打印搬移前目标矩阵 [CHECK1]
  buckyball.print %arrayB : memref<4x16xi8>
  
  // 使用mvin将数据从16字节对齐地址搬到暂存器
  // CHECK: mvin
  buckyball.bb_mvin %arrayA %spadAddr16 : memref<4x16xi8> i64
  // 使用mvout将数据从暂存器搬到16字节对齐的目标地址
  // CHECK: mvout
  buckyball.bb_mvout %arrayB %spadAddr16 : memref<4x16xi8> i64
  
  // 打印搬移后的输出矩阵 [CHECK2]
  buckyball.print %arrayB : memref<4x16xi8>
  
  // 释放分配的内存
  memref.dealloc %arrayB : memref<4x16xi8>
  return %0 : i8
}
