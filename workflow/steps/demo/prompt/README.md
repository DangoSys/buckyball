# Gemmini Ball Generator - 简化版自动生成系统

## 🎯 设计理念

**从复杂到简单**：抛弃多 Agent 协作的复杂架构，采用**单一智能 Agent**完成所有工作。

### 旧系统的问题
- ❌ 多个 Agent（spec_agent、code_agent、review_agent、master_agent）协作复杂
- ❌ Agent 之间的通信和状态同步容易出错
- ❌ 容易在某个 Agent 完成后停止，无法自动继续
- ❌ 错误恢复逻辑分散在多个 Agent 中

### 新系统的优势
- ✅ **单一 Agent**：一个 Agent 完成所有工作（学习、生成、编译、修复）
- ✅ **自动持续**：Agent 自动从 matmul → im2col → transpose → norm
- ✅ **智能修复**：编译失败自动分析错误并修复代码
- ✅ **简单直接**：没有复杂的 Agent 间通信协议

## 📁 文件结构

```
workflow/steps/demo/prompt/
├── README.md                    # 本文档
├── gemmini_task.md              # 任务描述
└── gemmini_ball_generator.md    # Agent 指令（系统 Prompt）

scripts/
└── run_gemmini_generator.sh     # 启动脚本

workflow/steps/demo/
└── simple_gemmini_agent.py      # Agent 执行引擎
```

## 🚀 使用方法

### 方式一：使用启动脚本（推荐）

```bash
cd /home/daiyongyuan/buckyball
bash scripts/run_gemmini_generator.sh
```

### 方式二：直接运行 Python 脚本

```bash
cd /home/daiyongyuan/buckyball/workflow/steps/demo
python3 simple_gemmini_agent.py
```

## 📋 工作流程

Agent 会自动执行以下步骤：

### 1. 学习阶段
- 读取参考代码（VecUnit.scala、VecBall.scala）
- 读取系统注册文件（DomainDecoder、busRegister、rsRegister、DISA）
- 理解 Ball 的结构和接口规范

### 2. 生成阶段（循环4次）
对于每个 Ball（matmul、im2col、transpose、norm）：

**2.1 创建目录**
```
arch/src/main/scala/prototype/gemmini/<ball>/
```

**2.2 生成代码**
- `<BallName>Unit.scala` - 主计算单元
- `<BallName>Ball.scala` - Ball 包装类

**2.3 更新系统注册**
- DomainDecoder.scala - 添加指令解码
- busRegister.scala - 实例化 Ball
- rsRegister.scala - 注册 Ball
- DISA.scala - 添加指令编码（如果缺失）

### 3. 验证阶段
**3.1 立即编译**
```bash
bash /home/daiyongyuan/buckyball/scripts/build_gemmini.sh build
```

**3.2 分析结果**
- ✅ 编译成功 → 继续下一个 Ball
- ❌ 编译失败 → 分析错误 → 修复代码 → 重新编译（最多5次）

### 4. 完成条件
- ✅ 所有 4 个 Ball 生成完成
- ✅ 所有代码能够编译成功
- ✅ 无编译错误

## 🛠️ Agent 可用工具

Agent 具有以下能力：

| 工具 | 功能 |
|-----|------|
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件内容 |
| `list_files` | 列出目录文件 |
| `make_dir` | 创建目录 |
| `run_build` | 运行编译脚本并分析结果 |
| `grep_files` | 搜索文件内容 |

## 📊 输出示例

```
==============================================================
Gemmini Ball Generator - 自动生成 4 个 Ball
==============================================================

[迭代 1]
🔧 执行 6 个工具调用
  - read_file({"path": "arch/src/main/scala/prototype/vector/VecUnit.scala"})
  - read_file({"path": "arch/src/main/scala/prototype/vector/VecBall.scala"})
  ...

[迭代 2]
🔧 执行 3 个工具调用
  - make_dir({"path": "arch/src/main/scala/prototype/gemmini/matmul"})
  - write_file({"path": "arch/.../MatMulUnit.scala", ...})
  - write_file({"path": "arch/.../MatMulBall.scala", ...})

[迭代 3]
🔧 执行 1 个工具调用
  - run_build({})
    ✅ 编译成功

✅ MATMUL Ball 完成！

[继续 im2col...]

==============================================================
执行总结
==============================================================
总迭代次数: 42
完成的 Ball: matmul, im2col, transpose, norm

✅ 任务成功完成！
```

## 🔧 配置

### 环境变量

在 `.env` 文件中配置（或使用默认值）：

```bash
API_BASE_URL=http://localhost:8000/v1
API_KEY=dummy-key
MODEL=qwen3-235b-a22b-instruct-2507
```

### 编译脚本

编译脚本位置：`/home/daiyongyuan/buckyball/scripts/build_gemmini.sh`
编译日志位置：`/home/daiyongyuan/buckyball/build_logs/gemmini_build.log`

## 🐛 故障排查

### 问题：Agent 停止执行
**原因**：可能达到最大迭代次数（100次）
**解决**：检查日志，如果需要可以增加 `max_iterations`

### 问题：编译一直失败
**原因**：代码错误无法自动修复
**解决**：
1. 查看编译日志：`/home/daiyongyuan/buckyball/build_logs/gemmini_build.log`
2. 手动修复代码
3. 重新运行 Agent

### 问题：API 调用失败
**原因**：LLM API 不可用或配置错误
**解决**：检查 `.env` 配置，确保 API 可访问

## 📝 修改 Agent 行为

### 调整 Agent 指令
编辑 `gemmini_ball_generator.md`：
- 修改代码生成策略
- 调整错误修复逻辑
- 添加新的约束条件

### 调整任务描述
编辑 `gemmini_task.md`：
- 修改 Ball 的顺序
- 添加新的 Ball
- 调整成功标准

## 🎓 设计原则

1. **简单优于复杂**：单一 Agent 而不是多 Agent 协作
2. **自动优于手动**：自动持续执行而不是分步骤等待
3. **修复优于报错**：自动修复错误而不是直接失败退出
4. **完整优于部分**：必须完成所有 Ball 才能停止

## 📚 参考

- 参考实现：`arch/src/main/scala/prototype/vector/`
- 系统注册：`arch/src/main/scala/examples/toy/balldomain/`
- Blink 接口：`framework/blink/`
- Chisel 文档：https://www.chisel-lang.org/

---

**版本**：2.0 - 简化版
**更新时间**：2025-11-10

