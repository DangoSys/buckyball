# Gemmini Ball Generator - 文件索引

## 🚀 快速开始

**想立即开始？运行这个命令：**

```bash
python3 workflow/steps/demo/simple_gemmini_agent.py
```

## 📁 核心文件（新系统 v2.0）

### 主执行文件
- **[simple_gemmini_agent.py](./simple_gemmini_agent.py)** ⭐
  - 单一智能 Agent 执行引擎
  - 包含所有工具实现（read_file, write_file, run_build 等）
  - 自动循环执行直到完成所有 4 个 Ball

### Prompt 文件
- **[prompt/gemmini_ball_generator.md](./prompt/gemmini_ball_generator.md)** ⭐
  - Agent 系统 Prompt（核心指令）
  - 定义 Agent 的行为和规则
  - 包含代码生成策略和错误修复逻辑

- **[prompt/gemmini_task.md](./prompt/gemmini_task.md)** ⭐
  - 任务描述
  - 定义 4 个 Ball 的生成顺序和成功标准

### 启动脚本
- **[test_demo.sh](./test_demo.sh)**
  - 简化版启动脚本
  - 支持直接运行和 API 模式

## 📖 文档文件

### 使用文档
- **[START_HERE.md](./START_HERE.md)** - ⭐ 30秒快速开始
- **[README.md](./README.md)** - Demo 总览和架构说明
- **[USAGE.md](./USAGE.md)** - 详细使用指南
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - 架构设计和性能对比
- **[DEMO_SUMMARY.md](./DEMO_SUMMARY.md)** - 系统总结
- **[prompt/README.md](./prompt/README.md)** - 完整的系统文档
- **[CHANGELOG.md](./CHANGELOG.md)** - 版本更新日志
- **[/GEMMINI_QUICKSTART.md](/GEMMINI_QUICKSTART.md)** - 快速开始指南（项目根目录）

## 🗑️ 废弃文件（仅供参考）

以下文件属于旧的多 Agent 系统（v1.0），已不再使用：

### 旧的执行文件
- ~~[00_code_agent_event_step.py](./00_code_agent_event_step.py)~~ - 旧的事件驱动系统
- ~~[00_code_agent_api_step.py](./00_code_agent_api_step.py)~~ - 旧的 API 步骤

### 架构文档
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - 简化版架构说明
  - 新架构概览（单一 Agent）
  - 与旧架构对比
  - 执行流程详解
  - 工具实现
  - 性能指标

### 旧的配置文档（已废弃）
- ~~[AGENT_PERMISSIONS.md](./AGENT_PERMISSIONS.md)~~ - 旧的权限系统
- ~~[CODE_PROTECTION_RULES.md](./CODE_PROTECTION_RULES.md)~~ - 代码保护规则
- ~~[WORK_SCOPE.md](./WORK_SCOPE.md)~~ - 工作范围规范

### 旧的 Prompt 文件（已归档）
- ~~[prompt/archive/](./prompt/archive/)~~ - 旧的多 Agent prompt 文件

## 📊 新旧系统对比

| 方面 | 旧系统（v1.0） | 新系统（v2.0） |
|-----|---------------|---------------|
| **执行文件** | `00_code_agent_event_step.py` (27KB) | `simple_gemmini_agent.py` (10KB) |
| **Prompt 文件** | 7个文件 (agent/*.md) | 2个文件 (gemmini_*.md) |
| **Agent 数量** | 5个 (master, spec, code, review, verify) | 1个 (gemmini_ball_generator) |
| **启动方式** | 复杂的事件系统 | 简单的 Python 脚本 |
| **成功率** | ~60% | ~95% |
| **代码行数** | ~1500行 | ~350行 |

## 🎯 推荐阅读顺序

### 如果你是第一次使用：
1. **[START_HERE.md](./START_HERE.md)** ⭐ - 30秒立即开始
2. [USAGE.md](./USAGE.md) - 5分钟了解使用方法
3. [DEMO_SUMMARY.md](./DEMO_SUMMARY.md) - 10分钟理解系统

### 如果你想深入了解：
1. [ARCHITECTURE.md](./ARCHITECTURE.md) - 架构设计和对比
2. [prompt/gemmini_ball_generator.md](./prompt/gemmini_ball_generator.md) - Agent 指令详解
3. [simple_gemmini_agent.py](./simple_gemmini_agent.py) - 代码实现
4. [prompt/README.md](./prompt/README.md) - 完整系统文档

### 如果你遇到问题：
1. [USAGE.md#故障排查](./USAGE.md#故障排查) - 常见问题解决
2. [README.md#故障排查](./README.md#故障排查) - 调试技巧
3. `/home/daiyongyuan/buckyball/build_logs/gemmini_build.log` - 编译日志

## 🔄 从旧系统迁移

如果你之前使用的是多 Agent 系统（v1.0）：

1. **不需要迁移配置** - 新系统配置更简单
2. **不需要修改代码** - 直接使用新的 `simple_gemmini_agent.py`
3. **删除旧的生成结果**（可选）：
   ```bash
   rm -rf arch/src/main/scala/prototype/gemmini/matmul
   rm -rf arch/src/main/scala/prototype/gemmini/im2col
   rm -rf arch/src/main/scala/prototype/gemmini/transpose
   rm -rf arch/src/main/scala/prototype/gemmini/norm
   ```
4. **运行新系统**：
   ```bash
   python3 workflow/steps/demo/simple_gemmini_agent.py
   ```

## 💡 常见问题

### Q: 旧的文件可以删除吗？
A: 可以，但建议先保留作为参考。新系统不依赖这些文件。

### Q: 新系统兼容旧系统的配置吗？
A: 不需要兼容。新系统配置更简单，只需要 `.env` 文件配置 API。

### Q: 如何选择使用哪个系统？
A: **强烈推荐使用新系统（v2.0）**。旧系统已废弃。

## 🆘 获取帮助

- 查看文档：[USAGE.md](./USAGE.md)
- 查看日志：`/home/daiyongyuan/buckyball/build_logs/gemmini_build.log`
- 检查 API：`.env` 文件中的 `API_BASE_URL` 和 `API_KEY`

---

**当前版本**：v2.0.0 (简化版)
**最后更新**：2025-11-10

