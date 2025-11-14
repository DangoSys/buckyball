# Gemmini NPU 任务列表

本目录包含多个 Gemmini NPU 相关的设计任务，涵盖从基础计算 Ball 到高级脉动阵列架构。

## 📋 任务总览

### 任务 1：基础计算 Ball（gemmini_npu.md）
**难度**：⭐⭐  
**目标**：自动生成 4 个基础计算 Ball
- MatMul（矩阵乘法）
- Im2col（图像到列转换）
- Transpose（矩阵转置）
- Norm（归一化）

**关键要求**：
- ✅ 代码自动生成
- ✅ 编译验证
- ✅ 系统集成

---

### 任务 2：ABFT 可靠性脉动阵列（task2_abft_systolic.md）
**难度**：⭐⭐⭐⭐  
**目标**：设计支持 WS/OS 数据流的脉动阵列，集成 ABFT 容错机制

**核心特性**：
- 🔄 **数据流模式**：Weight Stationary (WS) + Output Stationary (OS)
- 🛡️ **可靠性**：Algorithm-Based Fault Tolerance (ABFT)
  - 行/列校验和
  - 错误检测与定位
  - 错误恢复
- 📊 **阵列规格**：4×4，8-bit 数据，32-bit 累加器

**技术亮点**：
- 在线校验（计算过程中同步计算校验和）
- 交叉定位（精确到单个 PE 的错误）
- 冗余计算路径
- ABFT 开销 < 15%

**C Test**：
- 无错误矩阵乘法测试
- 错误注入与检测测试
- WS/OS 模式验证
- 性能开销测试

---

### 任务 3：可配置位宽脉动阵列（task3_configurable_systolic.md）
**难度**：⭐⭐⭐⭐⭐  
**目标**：设计支持 WS/OS 数据流的脉动阵列，支持运行时可配置位宽和量化精度

**核心特性**：
- 🔄 **数据流模式**：Weight Stationary (WS) + Output Stationary (OS)
- 🎚️ **可配置位宽**：2-bit / 4-bit / 8-bit / 16-bit（运行时切换）
- ⚖️ **量化支持**：
  - 对称量化（INT2/INT4/INT8/INT16）
  - 非对称量化（UINT8）
  - 可配置缩放因子和零点
- 📦 **位宽打包**：低位宽数据高效存储
- 📊 **阵列规格**：8×8（可配置 4/8/16），最大 16-bit

**技术亮点**：
- FlexiblePE：动态位宽 MAC 操作
- 位宽转换和打包/解包逻辑
- 重新量化和截断
- 混合精度计算流水线

**性能指标**：
- INT2 吞吐量 = INT16 的 8 倍
- INT4 吞吐量 = INT16 的 4 倍
- INT8 量化误差 < 0.5%

**C Test**：
- INT2/INT4/INT8/INT16 模式测试
- 对称/非对称量化测试
- 位宽打包验证
- 混合精度流水线
- 性能对比 benchmark

---

### 任务 4：三种数据流脉动阵列（task4_triple_dataflow_systolic.md）
**难度**：⭐⭐⭐⭐⭐⭐  
**目标**：设计支持 WS/OS/RS 三种数据流的脉动阵列

**核心特性**：
- 🔄 **三种数据流模式**：
  - **Weight Stationary (WS)**：权重固定，激活流动
  - **Output Stationary (OS)**：输出固定，权重和激活流动
  - **Row Stationary (RS)**：平衡所有数据重用，能效最优 ⭐
- 📊 **阵列规格**：16×16，16-bit 数据，32-bit 累加器
- 💾 **RS 模式特性**：
  - 每个 PE 有本地寄存器堆（64 entries）
  - Tile 调度和映射
  - 对角线数据流
  - 双缓冲和预取

**数据流对比**：

| 特性 | WS | OS | RS |
|------|----|----|-----|
| 权重重用 | ✅ 高 | ❌ 无 | ✅ 中 |
| 输入重用 | ❌ 无 | ❌ 无 | ✅ 中 |
| 输出重用 | ❌ 无 | ✅ 高 | ✅ 中 |
| DRAM 访问 | 中 | 高 | **✅ 最低** |
| 能效 | 中 | 低 | **✅ 最高** |
| PE 复杂度 | 低 | 低 | 高 |

**技术亮点**：
- TripleDataflowPE：统一的三模式 PE
- RS 模式的复杂 Tile 调度器
- 对角线数据流和 skew 机制
- 双缓冲和预取优化

**性能指标**：
- RS 模式 DRAM 访问 = WS 的 30-50%
- RS 模式能效 = WS 的 1.5-2.5 倍
- PE 利用率 > 85%

**C Test**：
- WS/OS/RS 三种模式功能测试
- 结果一致性验证
- 性能对比 benchmark
- 能耗估算对比
- Tile 调度验证

---

## 🎯 任务难度分析

### 任务 1（基础）：⭐⭐
- **主要挑战**：代码生成和系统集成
- **时间估计**：2-3 天
- **适合人群**：熟悉 Chisel 基础

### 任务 2（ABFT）：⭐⭐⭐⭐
- **主要挑战**：
  - ABFT 算法实现
  - 在线校验逻辑
  - 错误定位机制
- **时间估计**：1-2 周
- **适合人群**：理解容错计算原理

### 任务 3（可配置位宽）：⭐⭐⭐⭐⭐
- **主要挑战**：
  - 多位宽 PE 设计（最复杂）
  - 位宽打包/解包逻辑
  - 量化参数管理
  - 重新量化和截断
- **时间估计**：2-3 周
- **适合人群**：理解量化推理和硬件优化

### 任务 4（三数据流）：⭐⭐⭐⭐⭐⭐
- **主要挑战**：
  - RS 模式实现（**极高难度**）
  - Tile 调度算法
  - 对角线数据流时序控制
  - 三模式统一接口
  - 性能优化
- **时间估计**：3-4 周
- **适合人群**：深入理解脉动阵列架构和数据流优化
- **建议**：先完成任务 2，再尝试任务 4

---

## 📚 学习路径建议

### 路径 1：基础到进阶
```
任务 1 → 任务 2 → 任务 3 → 任务 4
(基础)   (容错)   (量化)   (数据流)
```

### 路径 2：数据流优化专精
```
任务 1 → 任务 2 → 任务 4
(基础)   (容错)   (三数据流)
```

### 路径 3：量化加速专精
```
任务 1 → 任务 3
(基础)   (可配置位宽)
```

---

## 🛠️ 使用方法

### 1. 选择任务
根据你的需求和技能水平选择合适的任务。

### 2. 阅读任务文档
每个任务的 `.md` 文件包含：
- 详细的技术规格
- Chisel 代码结构
- 状态机设计
- 完整的 C Test Workload
- 验证标准

### 3. 使用 Agent 自动生成
```bash
cd /home/daiyongyuan/buckyball
python3 workflow/steps/demo/simple_gemmini_agent.py --task task2_abft_systolic.md
```

或者手动指定任务提示词：
```bash
# 修改 simple_gemmini_agent.py 中的任务提示词路径
TASK_PROMPT = "workflow/steps/demo/prompt/task/task3_configurable_systolic.md"
```

### 4. 编译验证
```bash
bash scripts/build_gemmini.sh build
```

---

## 📖 参考资料

### 脉动阵列基础
- Kung, H. T., & Leiserson, C. E. (1980). "Systolic Arrays"

### ABFT
- Huang, K. H., & Abraham, J. A. (1984). "Algorithm-Based Fault Tolerance for Matrix Operations"

### 量化推理
- Jacob et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"

### Row Stationary
- Chen et al. (2016). "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs"

### Gemmini
- Genc et al. (2021). "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation via Full-Stack Integration"

---

## 📝 贡献指南

如果你完成了某个任务，欢迎贡献：
1. 实现代码（Chisel）
2. 测试代码（C）
3. 性能测试报告
4. 遇到的问题和解决方案

---

## ❓ 常见问题

**Q: 这些任务可以同时实现吗？**  
A: 可以，但建议按顺序实现。任务 1 是基础，其他任务在此基础上扩展。

**Q: RS 模式为什么这么复杂？**  
A: RS 模式需要在每个 PE 中实现本地寄存器堆和复杂的 Tile 调度逻辑，但它能实现最优的能效。

**Q: 可以只实现部分功能吗？**  
A: 可以。例如任务 3 可以先只实现 8-bit 和 16-bit，后续再扩展到 2-bit 和 4-bit。

**Q: 如何调试脉动阵列？**  
A: 建议：
1. 从小规模开始（4×4）
2. 使用波形查看器（GTKWave）
3. 打印中间结果
4. 编写详细的单元测试

---

**最后更新**：2025-01-13  
**维护者**：Gemmini NPU Team

