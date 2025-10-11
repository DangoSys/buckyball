# LayerNorm Ball

LayerNorm (Layer Normalization) 加速单元是BuckyBall框架中的专用计算加速器，用于高效执行层归一化运算。

## 功能特性

- **数据格式**: 支持INT8/INT32输入和输出
- **向量化处理**: 每次处理16个元素（veclane=16）
- **流水线架构**: 5级流水线（ID, Load, Reduce, Normalize, Store）
- **计算方法**: 计算均值、方差，执行归一化和可选的仿射变换
- **存储接口**: 支持Scratchpad和Accumulator读写
- **支持模式**: 可配置的归一化维度和可选的gamma/beta参数

## 数学定义

给定输入向量 x = [x₀, x₁, ..., x_{N-1}]：

1. **计算均值**: μ = (1/N) * Σᵢ xᵢ
2. **计算方差**: σ² = (1/N) * Σᵢ (xᵢ - μ)²
3. **归一化**: x̂ᵢ = (xᵢ - μ) / √(σ² + ε)
4. **仿射变换（可选）**: yᵢ = γᵢ * x̂ᵢ + βᵢ

其中：
- N 是归一化维度大小
- ε 是防止除零的小常数（典型值：1e-5）
- γ (gamma) 和 β (beta) 是可学习的仿射参数

## 模块结构

```
LayerNormBall (Ball包装器)
  └── LayerNormUnit (顶层模块)
      ├── LayerNormCtrlUnit (控制单元)
      ├── LayerNormLoadUnit (数据加载单元)
      ├── LayerNormReduceUnit (归约单元 - 计算均值和方差)
      ├── LayerNormNormalizeUnit (归一化和仿射变换单元)
      └── LayerNormStoreUnit (存储单元)
```

## 指令格式

LayerNorm指令通过RoCC接口下发，包含以下参数：

- `iter`: 批次迭代次数（1-1024）
- `op1_bank/op1_bank_addr`: 输入数据Bank和地址
- `wr_bank/wr_bank_addr`: 输出数据Bank和地址
- `is_acc`: 数据类型选择（0=SRAM/INT8, 1=ACC/INT32）
- `special[11:0]`: 归一化维度（向量数）
- `special[23:12]`: Gamma参数地址
- `special[35:24]`: Beta参数地址
- `special[37:36]`: 参数Bank号
- `special[38]`: 使能仿射变换

## 使用示例

详细的使用示例和软件接口请参考 [spec.md](./spec.md)。

## 测试

测试用例位于 `bb-tests/` 目录下。

运行测试：
```bash
bbdev verilator run <test_binary>
```

## 参考

- [spec.md](./spec.md) - 详细设计规范
- BuckyBall Framework Documentation
- Ball Domain Architecture Guide
