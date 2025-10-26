# 任务：用 BuckyBall 框架从头实现 Gemmini NPU（已精简为可自动执行的流程）

## 目标（精炼）

实现 Gemmini 的四个计算 Ball（MatMul、Im2col、Transpose、Norm）的**自动化生成与集成骨架**。生成内容覆盖：

- 每个 Ball 的 `spec.md`（若缺失，由 spec_agent 创建）
- 每个 Ball 的 Chisel 源代码骨架（.scala 文件）
- 必要的系统注册追加（DomainDecoder/BusRegister/RSRegister 的 append 脚本）

> 说明：本任务**不包含** DMA、scratchpad、accumulator 的底层实现（框架已提供）。本工程侧重于生成可被人工或自动补全的骨架与规范，使 AI 可继续完成 RTL 细节。

## 优先顺序

1. 为每个 Ball 生成 `spec.md`（如果缺失） —> spec_agent
2. 根据 spec 生成 `.scala` 骨架 —> code_agent
3. 将变更点和新增文件列表返回给 master_agent（以便 review/verify）

## 必要输出结构

```
arch/src/main/scala/prototype/gemmini/
├── matmul/
│   ├── spec.md
│   ├── MatMulUnit.scala
│   └── ...
├── im2col/
│   ├── spec.md
│   └── Im2colUnit.scala
├── transpose/
│   ├── spec.md
│   └── TransposeUnit.scala
└── norm/
    ├── spec.md
    └── NormUnit.scala
```

## ISA/Instruction 覆盖（摘要）

- 必须兼容 Gemmini 常用 funct（CONFIG_CMD/COMPUTE/PRELOAD 等），但实际指令解析在 DomainDecoder 中追加 BitPat（由 master_agent 负责追加到 decode 表）。
- code_agent 生成的 Unit 骨架应包含对 `funct` 字段的 switch-stub。

## 集成提示（给 AI 的直接执行说明）

- 若你是自动化脚本：先调用 spec_agent 生成缺失 spec；再调用 code_agent 生成骨架；最后调用 review_agent 进行静态审查。
- 生成文件一律使用 `.scala` 后缀，避免覆盖手写代码。

## 验证建议

- 提交文件列表后，由 review_agent 校验：是否存在 `spec.md`、Blink 接口声明、以及各单元的 IO stub。
- verify_agent 负责调用工作流 API 做仿真（不得由 code_agent 调用）。

---

**本文件为 Gemmini NPU 项目的操作与交付规范（已优化以方便 AI 直接生成文件骨架）。**
