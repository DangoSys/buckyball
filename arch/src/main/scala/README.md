# BuckyBall Scala 源码

该目录包含了 BuckyBall 项目的所有 Scala/Chisel 硬件描述语言源码，实现了硬件架构设计和仿真环境。

## 概述

BuckyBall 采用 Scala/Chisel 作为硬件描述语言，基于 Berkeley 的 Rocket-chip 和 Chipyard 框架构建。该目录包含了从底层硬件组件到系统级集成的实现。

主要功能模块包括：
- **framework**: 核心框架实现，包含处理器核心、内存子系统、总线互连等
- **prototype**: 专用加速器的原型实现
- **examples**: 示例配置和参考设计
- **sims**: 仿真环境配置和接口
- **Util**: 通用工具类和辅助函数

## 代码结构

```
scala/
├── framework/          - BuckyBall 核心框架
│   ├── blink/          - Blink 通信组件
│   ├── builtin/        - 内置硬件组件
│   │   ├── frontend/   - 前端处理组件
│   │   ├── memdomain/  - 内存域实现
│   │   └── util/       - 框架工具类
│   └── rocket/         - Rocket 核心扩展
├── prototype/          - 专用加速器原型
│   ├── format/         - 数据格式处理
│   ├── im2col/         - 图像处理加速
│   ├── matrix/         - 矩阵运算引擎
│   ├── transpose/      - 矩阵转置加速
│   └── vector/         - 向量处理单元
├── examples/           - 示例和配置
│   └── toy/            - 玩具示例系统
├── sims/               - 仿真配置
│   ├── firesim/        - FireSim FPGA 仿真
│   └── verilator/      - Verilator 仿真
└── Util/               - 通用工具类
```

## 模块说明

### framework/ - 核心框架
实现了 BuckyBall 的核心架构组件，包括：
- 处理器核心和扩展
- 内存子系统和缓存层次
- 总线互连和通信协议
- 系统配置和参数化机制

### prototype/ - 加速器原型
包含专用计算加速器的硬件实现：
- 机器学习加速器（矩阵运算、卷积等）
- 数据处理加速器（格式转换、转置等）
- 向量处理单元（SIMD、多线程等）

### examples/ - 示例配置
提供系统配置示例和参考设计：
- 基础配置模板
- 自定义扩展示例
- 集成测试用例

### sims/ - 仿真环境
支持多种仿真器和验证环境：
- Verilator 仿真
- FireSim FPGA 仿真
- 性能分析和调试工具

## 开发指南

### 构建系统
BuckyBall 使用 Mill 作为构建工具：
```bash
# 编译所有模块
mill arch.compile

# 生成 Verilog
mill arch.runMain examples.toy.ToyBuckyBall

# 运行测试
mill arch.test
```

### 代码规范
- 遵循 Scala 和 Chisel 编码规范
- 使用 ScalaFmt 进行代码格式化
- 每个模块包含文档和测试
- 配置参数化使用 Chipyard Config 系统

### 扩展开发
1. **新增加速器**: 在 prototype/ 目录下创建新模块
2. **修改框架**: 在 framework/ 目录下扩展现有组件
3. **添加配置**: 在 examples/ 目录下创建新的配置文件
4. **集成测试**: 使用 sims/ 目录下的仿真环境验证

## 相关文档

- [框架核心文档](framework/README.md)
- [加速器原型文档](prototype/README.md)
- [示例配置文档](examples/README.md)
- [仿真环境文档](sims/README.md)
- [工具类文档](Util/README.md)
