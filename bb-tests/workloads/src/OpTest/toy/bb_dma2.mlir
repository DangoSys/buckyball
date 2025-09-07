// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec: 
// 目的：验证dma模块快速交替读写的正确性
// 1. 打印输入矩阵A和B
// 2. 打印搬移前目标地址的矩阵  [CHECK1] 打印结果应为全0矩阵
// 3. 执行快速交替操作：逐行使用mvin读取，mvout写入
// 4. 打印搬移后目标地址的矩阵  [CHECK2] 打印结果应该显示A和B内容交换

memref.global "private" @input_matrix_a : memref<2x16xi8> = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                                                                   [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]>

memref.global "private" @input_matrix_b : memref<2x16xi8> = dense<[[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
                                                                                    [116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 126, 125, 124, 123]]>

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  %spadAddr1 = arith.constant 10 : i64  // 暂存器地址1
  %spadAddr2 = arith.constant 50 : i64  // 暂存器地址2
  
  %arrayA = memref.get_global @input_matrix_a : memref<2x16xi8>
  %arrayB = memref.get_global @input_matrix_b : memref<2x16xi8>
  %arrayTemp = memref.alloc() {alignment = 16} : memref<2x16xi8>
  
  // 打印输入矩阵A和B
  buckyball.print %arrayA : memref<2x16xi8>
  buckyball.print %arrayB : memref<2x16xi8>
  // 打印搬移前临时矩阵 [CHECK1]
  buckyball.print %arrayTemp : memref<2x16xi8>
  
  // 快速交替mvin/mvout操作序列
  // 第一步：A -> 暂存器1
  // CHECK: mvin
  buckyball.bb_mvin %arrayA %spadAddr1 : memref<2x16xi8> i64
  // 第二步：暂存器1 -> temp
  // CHECK: mvout
  buckyball.bb_mvout %arrayTemp %spadAddr1 : memref<2x16xi8> i64
  
  // 第三步：B -> 暂存器2  
  // CHECK: mvin
  buckyball.bb_mvin %arrayB %spadAddr2 : memref<2x16xi8> i64
  // 第四步：暂存器2 -> A
  // CHECK: mvout
  buckyball.bb_mvout %arrayA %spadAddr2 : memref<2x16xi8> i64
  
  // 第五步：temp -> 暂存器1
  // CHECK: mvin
  buckyball.bb_mvin %arrayTemp %spadAddr1 : memref<2x16xi8> i64
  // 第六步：暂存器1 -> B
  // CHECK: mvout
  buckyball.bb_mvout %arrayB %spadAddr1 : memref<2x16xi8> i64
  
  // 打印交换后的矩阵 [CHECK2]
  buckyball.print %arrayA : memref<2x16xi8>
  buckyball.print %arrayB : memref<2x16xi8>
  
  // 释放分配的内存
  memref.dealloc %arrayTemp : memref<2x16xi8>
  return %0 : i8
} 