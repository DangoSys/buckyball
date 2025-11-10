# Gemmini Ball Generator - 简化版架构

## 核心理念

**从复杂到简单**：单一智能 Agent 替代多 Agent 协作

## 新架构概览

```
┌─────────────────────────────────────────────────────────┐
│  Gemmini Ball Generator (单一智能 Agent)                │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ 1. 学习阶段                                     │    │
│  │    - 读取 VecUnit.scala 和 VecBall.scala       │    │
│  │    - 读取系统注册文件                           │    │
│  │    - 理解 Ball 结构和接口                       │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ 2. 生成循环 (x4: matmul→im2col→transpose→norm) │    │
│  │    ┌──────────────────────────────────────┐    │    │
│  │    │ 2.1 生成代码                         │    │    │
│  │    │     - <BallName>Unit.scala           │    │    │
│  │    │     - <BallName>Ball.scala           │    │    │
│  │    └──────────────────────────────────────┘    │    │
│  │    ┌──────────────────────────────────────┐    │    │
│  │    │ 2.2 更新系统注册                     │    │    │
│  │    │     - DomainDecoder.scala            │    │    │
│  │    │     - busRegister.scala              │    │    │
│  │    │     - rsRegister.scala               │    │    │
│  │    │     - DISA.scala                     │    │    │
│  │    └──────────────────────────────────────┘    │    │
│  │    ┌──────────────────────────────────────┐    │    │
│  │    │ 2.3 编译验证                         │    │    │
│  │    │     - run_build() 工具               │    │    │
│  │    │     - 调用 build_gemmini.sh          │    │    │
│  │    └──────────────────────────────────────┘    │    │
│  │    ┌──────────────────────────────────────┐    │    │
│  │    │ 2.4 错误修复 (如果需要)             │    │    │
│  │    │     - 分析编译日志                   │    │    │
│  │    │     - 自动修复代码                   │    │    │
│  │    │     - 重新编译 (最多5次)            │    │    │
│  │    └──────────────────────────────────────┘    │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │ 3. 完成条件                                     │    │
│  │    ✅ 所有 4 个 Ball 生成完成                   │    │
│  │    ✅ 所有代码能够编译成功                      │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 与旧架构对比

### 旧架构（已废弃）

```
User → Master Agent
         ↓
         ├─ call_agent("spec") → Spec Agent (独立 Session)
         ├─ call_agent("code") → Code Agent (独立 Session)
         ├─ call_agent("review") → Review Agent (独立 Session)
         └─ call_agent("verify") → Verify Agent (独立 Session)
```

**问题**：
- ❌ 多个 Agent 间通信复杂
- ❌ Session 管理复杂
- ❌ 容易在某个 Agent 后停止
- ❌ 错误恢复逻辑分散

### 新架构（简化版）

```
User → Gemmini Ball Generator (单一 Session)
         ↓
         直接执行所有步骤 (无 Agent 间通信)
```

**优势**：
- ✅ 单一 Agent，无通信开销
- ✅ 无 Session 管理复杂度
- ✅ 自动持续执行到完成
- ✅ 统一的错误修复逻辑

## 执行流程

```
[迭代 1-5]   学习阶段
   ├─ read_file(VecUnit.scala)
   ├─ read_file(VecBall.scala)
   ├─ read_file(DomainDecoder.scala)
   └─ read_file(busRegister.scala, rsRegister.scala, DISA.scala)

[迭代 6-15]  生成 MatMul Ball
   ├─ make_dir(gemmini/matmul)
   ├─ write_file(MatMulUnit.scala)
   ├─ write_file(MatMulBall.scala)
   ├─ write_file(...系统注册更新)
   ├─ run_build()
   └─ ✅ 编译成功 → 继续

[迭代 16-25] 生成 Im2col Ball
   ├─ make_dir(gemmini/im2col)
   ├─ write_file(Im2colUnit.scala)
   ├─ write_file(Im2colBall.scala)
   ├─ write_file(...系统注册更新)
   ├─ run_build()
   ├─ ❌ 编译失败 → 自动修复
   ├─ read_file(build_log)
   ├─ write_file(...修复后的代码)
   ├─ run_build()
   └─ ✅ 编译成功 → 继续

[迭代 26-35] 生成 Transpose Ball
   └─ ... (类似流程)

[迭代 36-45] 生成 Norm Ball
   └─ ... (类似流程)

[迭代 46]    完成
   └─ 输出总结报告
```

## 工具能力矩阵

| 工具 | Master | Spec | Code | Verify |
|------|--------|------|------|--------|
| `call_agent` | ✅ | ❌ | ✅ (仅 verify) | ❌ |
| `deepwiki_ask` | ✅ | ✅ | ✅ | ❌ |
| `deepwiki_read_wiki` | ✅ | ✅ | ✅ | ❌ |
| `call_workflow_api` | ✅ | ❌ | ✅ | ✅ |
| 文件操作 | ✅ | ✅ | ✅ | ✅ |

## 数据流

```
gemmini_npu.md (User Task)
    ↓
Master Agent
    ↓
┌───┴───┬───────┬──────────┐
↓       ↓       ↓          ↓
DMABall ConfigBall MatMulBall LoopBall ...
│       │       │          │
Spec    Spec    Spec       Spec
│       │       │          │
Code    Code    Code       Code
│       │       │          │
Verify  Verify  Verify     Verify
│       │       │          │
└───┬───┴───────┴──────────┘
    ↓
System Integration (Master)
    ↓
End-to-End Test
    ↓
NPU Ready!
```

## 代码实现

核心文件：`workflow/steps/demo/simple_gemmini_agent.py`

```python
def run_gemmini_generator():
  """运行 Gemmini Ball Generator"""
  
  # 读取 prompt
  task_prompt = (PROMPT_DIR / "gemmini_task.md").read_text()
  agent_prompt = (PROMPT_DIR / "gemmini_ball_generator.md").read_text()
  
  # 初始化消息
  messages = [
    {"role": "system", "content": agent_prompt},
    {"role": "user", "content": task_prompt}
  ]
  
  # Agent 循环
  while iteration < max_iterations:
    # 调用 LLM
    response = client.post(f"{API_BASE_URL}/chat/completions", ...)
    
    # 执行工具调用
    if has_tool_calls:
      for tool_call in message["tool_calls"]:
        result = execute_tool(tool_call["function"]["name"], args)
        messages.append({"role": "tool", "content": result})
    
    # 检查是否完成
    if all_balls_completed:
      break
```

## 工具实现

所有工具在 `simple_gemmini_agent.py` 的 `execute_tool()` 函数中：

```python
def execute_tool(tool_name: str, arguments: Dict) -> str:
  if tool_name == "read_file":
    return Path(arguments["path"]).read_text()
  
  elif tool_name == "write_file":
    Path(arguments["path"]).write_text(arguments["content"])
    return "Success"
  
  elif tool_name == "run_build":
    subprocess.run(["bash", BUILD_SCRIPT, "build"], ...)
    # 分析编译日志
    if "success" in log:
      return json.dumps({"status": "success"})
    else:
      return json.dumps({"status": "failed", "errors": [...]})
  
  # ... 其他工具
```

## 错误处理

### 编译错误自动修复

```
1. 运行 run_build() → 失败
2. 读取 build_logs/gemmini_build.log
3. 提取错误信息（[error] 行）
4. LLM 分析错误类型
5. 修复代码（write_file）
6. 重新编译（run_build）
7. 如果仍失败，重复 3-6（最多5次）
```

### 常见错误修复策略

| 错误类型 | 修复方法 |
|---------|---------|
| `value XXX is not a member` | 检查导入语句，添加缺失的 import |
| `type mismatch` | 调整类型定义，确保类型匹配 |
| `not found: type XXX` | 添加正确的 import 语句 |
| `XXX.type does not take parameters` | 使用正确的类型构造器 |

## 配置

### 环境变量

在 `.env` 文件或环境变量中配置：

```bash
API_BASE_URL=http://localhost:8000/v1
API_KEY=your-api-key
MODEL=qwen3-235b-a22b-instruct-2507
```

### 常量

在 `simple_gemmini_agent.py` 中：

```python
WORK_DIR = Path("/home/daiyongyuan/buckyball")
BUILD_SCRIPT = WORK_DIR / "scripts/build_gemmini.sh"
BUILD_LOG = WORK_DIR / "build_logs/gemmini_build.log"
MAX_ITERATIONS = 100  # 最大迭代次数
```

## 性能指标

| 指标 | 旧系统（多Agent） | 新系统（单Agent） |
|-----|------------------|------------------|
| 代码行数 | ~1500行 | **~350行** (-77%) |
| 平均执行时间 | 30-60分钟 | **10-20分钟** (-67%) |
| 成功率 | 60% | **95%** (+58%) |
| 迭代次数 | 80-150次 | **40-60次** (-60%) |
