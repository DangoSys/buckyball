// RUN: buddy-opt %s \
// RUN:     -lower-buckyball | \
// RUN: FileCheck %s

// Spec: 
// 目的：验证dma模块长读入读出的正确性
// 1. 动态生成大型输入矩阵，数据0~127反复填入
// 2. 打印搬移前目标地址的矩阵  [CHECK1] 打印结果应为全0矩阵
// 3. 使用mvin将大数据量从内存读入暂存器
// 4. 使用mvout将大数据量从暂存器写出到内存
// 5. 打印搬移后目标地址的矩阵  [CHECK2] 打印结果应该与输入矩阵相同

func.func @main() -> i8 {
  %0 = arith.constant 0 : i8
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c127 = arith.constant 127 : index
  %spadAddr = arith.constant 0 : i64      // 暂存器起始地址
  
  // ========== 行数配置区域（只需修改这里） ==========
  %total_rows = arith.constant 1024 : index        // 总行数
  %last_rows_start = arith.constant 1022 : index   // 最后两行起始 = total_rows - 2
  // %offset_elements = arith.constant 16336 : index  // 偏移 = last_rows_start × 16
  %offset_elements = arith.constant 16352 : index  // 偏移 = last_rows_start × 16
  // ================================================
  
  // 分配大型矩阵
  %arrayA = memref.alloc() {alignment = 16} : memref<1023x16xi8>
  %arrayB = memref.alloc() {alignment = 16} : memref<1023x16xi8>
  
  // 使用linalg.fill初始化arrayB为0
  linalg.fill ins(%0 : i8) outs(%arrayB : memref<1023x16xi8>)
  
  // 动态填充arrayA：0~127反复填入
  scf.for %i = %c0 to %total_rows step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %row_offset = arith.muli %i, %c16 : index
      %linear_idx = arith.addi %row_offset, %j : index
      %mod_val = arith.remui %linear_idx, %c127 : index
      %val = arith.index_cast %mod_val : index to i8
      memref.store %val, %arrayA[%i, %j] : memref<1023x16xi8>
    }
  }
  
  // 打印输入矩阵的前两行和最后两行
  %arrayA_head = memref.subview %arrayA[0, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1]>>
  %arrayA_tail = memref.subview %arrayA[1021, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1], offset: 16336>>
  buckyball.print %arrayA_head : memref<2x16xi8, strided<[16, 1]>>
  buckyball.print %arrayA_tail : memref<2x16xi8, strided<[16, 1], offset: 16336>>
  
  // 打印搬移前目标矩阵的前两行和最后两行 [CHECK1]
  %arrayB_head = memref.subview %arrayB[0, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1]>>
  %arrayB_tail = memref.subview %arrayB[1021, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1], offset: 16336>>
  buckyball.print %arrayB_head : memref<2x16xi8, strided<[16, 1]>>
  buckyball.print %arrayB_tail : memref<2x16xi8, strided<[16, 1], offset: 16336>>
  
  // 执行长读入读出操作
  // 第一步：长时间读入 - 将大数据量从内存读入暂存器
  // CHECK: mvin
  buckyball.bb_mvin %arrayA %spadAddr : memref<1023x16xi8> i64
  
  // 第二步：长时间读出 - 将大数据量从暂存器写出到内存
  // CHECK: mvout
  buckyball.bb_mvout %arrayB %spadAddr : memref<1023x16xi8> i64
  
  // 打印搬移后输出矩阵的前两行和最后两行 [CHECK2]
  %arrayB_head_after = memref.subview %arrayB[0, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1]>>
  %arrayB_tail_after = memref.subview %arrayB[1021, 0] [2, 16] [1, 1] : memref<1023x16xi8> to memref<2x16xi8, strided<[16, 1], offset: 16336>>
  buckyball.print %arrayB_head_after : memref<2x16xi8, strided<[16, 1]>>
  buckyball.print %arrayB_tail_after : memref<2x16xi8, strided<[16, 1], offset: 16336>>
  
  // 释放分配的内存
  memref.dealloc %arrayA : memref<1023x16xi8>
  memref.dealloc %arrayB : memref<1023x16xi8>
  return %0 : i8
} 