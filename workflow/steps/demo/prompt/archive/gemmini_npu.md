# 任务：用 BuckyBall 框架从头实现 Gemmini NPU（已精简为可自动执行的流程）

## 目标（精炼）

实现 Gemmini 的四个计算 Ball（Im2col）的**自动化生成与集成骨架**。生成内容覆盖：

- 每个 Ball 的 `spec.md`（若缺失，由 spec_agent 创建）
- 每个 Ball 的 Chisel 源代码骨架（.scala 文件）
- 必要的系统注册追加（DomainDecoder/BusRegister/RSRegister 的 append 脚本）

> 说明：本任务**不包含** DMA、scratchpad、accumulator 的底层实现（框架已提供）。本工程侧重于生成可被人工或自动补全的骨架与规范，使 AI 可继续完成 RTL 细节。

## 优先顺序


**开始执行指令：**
1. **立即开始**：不要等待，不要检查， 调用 spec_agent
2. **第一步**：调用 spec_agent 生成 spec.md
3. **检测并继续**：spec_agent完成后立即调用 code_agent，依次类推

对于Ball，按以下顺序执行：

1. **首先 生成 spec.md**（如果缺失，由 spec_agent 创建）
2. **根据 spec 生成所有必需文件** —> code_agent：
   - 必须生成 `<BallName>Unit.scala`（主计算单元）
   - 必须生成 `<BallName>Ball.scala`（Ball 包装类）
   - 必须追加系统注册代码（DomainDecoder、busRegister、rsRegister、DISA）
   - **不能只生成部分文件就停止**
3. **立即编译验证**：code_agent 必须立即调用 `bash /home/daiyongyuan/buckyball/scripts/build_gemmini.sh build`
4. **自动修复错误**：如果编译失败，code_agent 必须立即读取 `/home/daiyongyuan/buckyball/build_logs/gemmini_build.log` 并自动修复
5. **循环重试**：修复后重新编译，直到编译成功（最多重试5次）
6. **返回 JSON 格式**：code_agent 必须返回包含 `created_files`、`compilation_status` 的 JSON 格式

所有 Ball 完成后：

11. **最终全局编译验证**：**必须**运行 `/home/daiyongyuan/buckyball/scripts/build_gemmini.sh build` 验证所有代码可以编译
12. 如果编译失败，分析错误并修复，重复步骤 2-11
13. **只有编译成功后才能停止**

**⚠️ 严格执行：**
- 不能在完成某个 Ball 后停止
- 不能在代码生成后停止
- **不能只生成部分文件就停止**（必须生成 Unit.scala 和 Ball.scala）
- **不能生成文件后不调用编译脚本**
- **不能只返回文本说明而不返回 JSON 格式**
- **不能在编译失败后停止，必须立即自动修复**
- **必须**完成全部流程后才能停止

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
