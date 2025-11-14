# Gemmini NPU 任务快速导航

## 🎯 任务概览

本项目提供 **1 个基础任务** 和 **3 个高级任务**，涵盖从基础计算单元到复杂脉动阵列架构的完整设计。

---

## 📋 任务列表

### ✅ 任务 1：基础计算 Ball（必做）
**文件**：[`prompt/gemmini_task.md`](prompt/gemmini_task.md)  
**难度**：⭐⭐  
**时间**：2-3 天

**目标**：自动生成 4 个基础计算 Ball
- MatMul（矩阵乘法）
- Im2col（图像到列转换）
- Transpose（矩阵转置）
- Norm（归一化）

**特点**：
- 代码自动生成
- 编译自动验证
- 错误自动修复

**运行方式**：
```bash
cd /home/daiyongyuan/buckyball
bash scripts/run_gemmini_generator.sh
```

---

### 🛡️ 任务 2：ABFT 可靠性脉动阵列
**文件**：[`prompt/task/task2_abft_systolic.md`](prompt/task/task2_abft_systolic.md)  
**难度**：⭐⭐⭐⭐  
**时间**：1-2 周

**核心特性**：
- 🔄 WS/OS 双数据流模式
- 🛡️ ABFT（Algorithm-Based Fault Tolerance）
  - 行/列校验和
  - 在线错误检测
  - 精确错误定位（到单个 PE）
  - 自动错误恢复

**技术指标**：
- 4×4 脉动阵列
- 8-bit 数据，32-bit 累加器
- ABFT 开销 < 15%
- 单比特错误检测率 100%

**适用场景**：
- 航天器计算
- 医疗设备
- 安全关键系统

**关键挑战**：
- ABFT 算法实现
- 在线校验逻辑
- 错误注入和验证

---

### ⚖️ 任务 3：可配置位宽脉动阵列
**文件**：[`prompt/task/task3_configurable_systolic.md`](prompt/task/task3_configurable_systolic.md)  
**难度**：⭐⭐⭐⭐⭐  
**时间**：2-3 周

**核心特性**：
- 🔄 WS/OS 双数据流模式
- 🎚️ 运行时可配置位宽
  - 2-bit（INT2）
  - 4-bit（INT4）
  - 8-bit（INT8/UINT8）
  - 16-bit（INT16）
- ⚖️ 量化支持
  - 对称量化
  - 非对称量化
  - 可配置缩放因子和零点
- 📦 位宽打包优化

**技术指标**：
- 8×8 脉动阵列（可配置 4/8/16）
- INT2 吞吐量 = INT16 的 8 倍
- INT4 吞吐量 = INT16 的 4 倍
- INT8 量化误差 < 0.5%

**适用场景**：
- 量化神经网络推理
- 边缘设备部署
- 移动端 AI 加速

**关键挑战**：
- FlexiblePE 设计（最复杂）
- 位宽打包/解包逻辑
- 混合精度流水线
- 重新量化和截断

---

### 🚀 任务 4：三数据流脉动阵列（最难）
**文件**：[`prompt/task/task4_triple_dataflow_systolic.md`](prompt/task/task4_triple_dataflow_systolic.md)  
**难度**：⭐⭐⭐⭐⭐⭐  
**时间**：3-4 周

**核心特性**：
- 🔄 三种数据流模式
  - **Weight Stationary (WS)**：权重固定
  - **Output Stationary (OS)**：输出固定
  - **Row Stationary (RS)**：平衡重用，**能效最优** ⭐
- 💾 RS 模式特性
  - 每个 PE 配备本地寄存器堆（64 entries）
  - 复杂的 Tile 调度器
  - 对角线数据流
  - 双缓冲和预取

**数据流对比**：

| 特性 | WS | OS | RS |
|------|----|----|-----|
| 权重重用 | ✅ 高 | ❌ 无 | ✅ 中 |
| 输入重用 | ❌ 无 | ❌ 无 | ✅ 中 |
| 输出重用 | ❌ 无 | ✅ 高 | ✅ 中 |
| **DRAM 访问** | 中 | 高 | **✅ 最低（30-50%）** |
| **能效** | 中 | 低 | **✅ 最高（1.5-2.5x）** |
| PE 复杂度 | 低 | 低 | 高 |

**技术指标**：
- 16×16 脉动阵列
- 16-bit 数据，32-bit 累加器
- RS 模式 DRAM 访问 = WS 的 30-50%
- RS 模式能效 = WS 的 1.5-2.5 倍

**适用场景**：
- 大规模神经网络训练
- 能效敏感场景
- 数据中心 AI 加速

**关键挑战**：
- RS 模式实现（**极高难度**）
- Tile 调度算法
- 对角线数据流时序控制
- 三模式统一接口

---

## 🎓 学习路径

### 路径 1：循序渐进（推荐）
```
任务 1 → 任务 2 → 任务 3 → 任务 4
(基础)   (容错)   (量化)   (数据流)
  ↓        ↓        ↓        ↓
 必做     中级     高级    专家级
```

### 路径 2：数据流优化专精
```
任务 1 → 任务 2 → 任务 4
(基础)   (容错)   (三数据流)
```
适合：对脉动阵列数据流和能效优化感兴趣

### 路径 3：量化加速专精
```
任务 1 → 任务 3
(基础)   (可配置位宽)
```
适合：对量化神经网络和边缘部署感兴趣

---

## 📊 难度对比

| 任务 | 难度 | 代码量 | Chisel 复杂度 | C Test 复杂度 | 调试难度 |
|------|------|--------|-------------|-------------|---------|
| 任务 1 | ⭐⭐ | ~500 行 | 低 | 低 | 低 |
| 任务 2 | ⭐⭐⭐⭐ | ~1500 行 | 中 | 中 | 中 |
| 任务 3 | ⭐⭐⭐⭐⭐ | ~2500 行 | **高** | 高 | 高 |
| 任务 4 | ⭐⭐⭐⭐⭐⭐ | ~3500 行 | **极高** | 高 | **极高** |

---

## 🛠️ 快速开始

### 1. 任务 1（基础）
```bash
cd /home/daiyongyuan/buckyball
bash scripts/run_gemmini_generator.sh
```

### 2. 任务 2-4（高级）
```bash
# 编辑 simple_gemmini_agent.py，修改任务提示词路径
# TASK_PROMPT = "workflow/steps/demo/prompt/task/task2_abft_systolic.md"

python3 workflow/steps/demo/simple_gemmini_agent.py
```

---

## 📚 详细文档

- **任务总览**：[`prompt/task/README.md`](prompt/task/README.md)
- **系统说明**：[`prompt/README.md`](prompt/README.md)
- **使用指南**：[`USAGE.md`](USAGE.md)
- **演示总结**：[`DEMO_SUMMARY.md`](DEMO_SUMMARY.md)

---

## 🎯 推荐顺序

### 初学者
1. ✅ 先完成**任务 1**（必做）
2. 📖 阅读任务 2-4 的文档，理解概念
3. 🎯 选择一个感兴趣的任务深入学习

### 中级
1. ✅ 完成任务 1
2. ✅ 完成任务 2（理解 ABFT 容错机制）
3. 🎯 根据兴趣选择任务 3 或 4

### 高级
1. ✅ 完成所有任务
2. 🔬 优化性能和能效
3. 📝 编写详细的技术报告
4. 🎤 分享实现经验

---

## ⚠️ 注意事项

### 任务 3 特别提醒
- FlexiblePE 设计是最复杂的部分
- 位宽打包逻辑容易出错，需要详细测试
- 建议先从 8-bit 和 16-bit 开始，再扩展到 2-bit 和 4-bit

### 任务 4 特别提醒
- RS 模式实现难度极高
- 建议从小规模（4×4）开始验证
- 使用波形图（GTKWave）仔细检查时序
- Tile 调度器逻辑复杂，需要详细的单元测试
- 建议先实现并验证 WS 和 OS 模式，再挑战 RS 模式

---

## 🤝 贡献

如果你完成了某个任务，欢迎分享：
1. 实现代码（Chisel）
2. 测试代码（C）
3. 性能测试报告
4. 遇到的问题和解决方案
5. 优化经验和技巧

---

## 📖 参考资料

### ABFT（任务 2）
- Huang, K. H., & Abraham, J. A. (1984). "Algorithm-Based Fault Tolerance for Matrix Operations"

### 量化推理（任务 3）
- Jacob et al. (2018). "Quantization and Training of Neural Networks"
- Rastegari et al. (2016). "XNOR-Net: ImageNet Classification Using Binary CNNs"

### Row Stationary（任务 4）
- Chen et al. (2016). "Eyeriss: An Energy-Efficient Reconfigurable Accelerator"
- Chen et al. (2017). "Understanding Reuse, Performance, and Hardware Cost of DNN Dataflows"

### Gemmini
- Genc et al. (2021). "Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation"

---

**最后更新**：2025-01-13  
**维护者**：Gemmini NPU Team

**祝你学习愉快！🚀**

