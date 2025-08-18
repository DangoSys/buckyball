# Build Verilator Workflow

这是一个基于Motia的事件驱动构建工作流，用于重写原有的Makefile构建逻辑。该工作流支持清理、Verilog生成、Verilator编译和仿真运行的完整流程。

## 工作流程概览

```
API调用 → 清理(可选) → Verilog生成 → Verilator编译 → 仿真运行 → 完成
```

## 使用方法

### 基本用法

```bash
# 运行完整的构建和仿真流程
npx motia emit --topic "build-verilator" --message '{"data": {}}'

# 仅清理构建目录
npx motia emit --topic "build-verilator" --message '{"data": {"target": "clean"}}'

# 仅生成Verilog代码
npx motia emit --topic "build-verilator" --message '{"data": {"target": "verilog"}}'

# 跳过清理步骤直接构建
npx motia emit --topic "build-verilator" --message '{"data": {"clean": false}}'
```

### 参数配置

支持的参数：

- `debug` (boolean): 是否启用调试模式，默认为 `false`
- `workload` (string): 指定仿真workload文件路径，默认使用内置workload
- `clean` (boolean): 是否在构建前清理，默认为 `true`
- `target` (string): 构建目标，可选值：
  - `"run"` (默认): 运行完整流程
  - `"sim"`: 同run，运行完整流程
  - `"verilog"`: 仅生成Verilog代码
  - `"clean"`: 仅清理构建目录

### 使用示例

#### 1. 调试模式运行

```bash
npx motia emit --topic "build-verilator" --message '{
  "data": {
    "debug": true,
    "target": "run"
  }
}'
```

#### 2. 指定自定义workload

```bash
npx motia emit --topic "build-verilator" --message '{
  "data": {
    "workload": "custom_test_workload",
    "debug": false
  }
}'
```

#### 3. 快速重建（跳过清理）

```bash
npx motia emit --topic "build-verilator" --message '{
  "data": {
    "clean": false,
    "target": "run"
  }
}'
```

#### 4. 仅生成Verilog并查看输出

```bash
npx motia emit --topic "build-verilator" --message '{
  "data": {
    "target": "verilog",
    "debug": true
  }
}'
```

## 工作流步骤详解

### 1. API入口 (`01_build_verilator_api_step.py`)
- 接收HTTP POST请求
- 解析构建参数
- 根据target和clean参数决定触发哪个事件

### 2. 清理步骤 (`02_clean_event_step.py`)
- 清理 `arch/build` 目录
- 为新的构建准备干净的环境
- 自动触发下一步Verilog生成

### 3. Verilog生成 (`03_verilog_generation_event_step.py`)
- 使用mill工具生成Verilog代码
- 支持调试模式优化选项
- 生成的文件保存在 `arch/build` 目录

### 4. Verilator编译 (`04_verilator_build_event_step.py`)
- 编译Verilog代码为C++可执行文件
- 处理所有必要的include路径和链接库
- 生成的可执行文件位于 `arch/build/obj_dir/VTestHarness`

### 5. 仿真运行 (`05_simulation_event_step.py`)
- 运行Verilator仿真
- 支持自定义workload
- 保存仿真日志到 `arch/log` 目录

### 6. 错误处理 (`06_error_handler_event_step.py`)
- 统一处理各步骤的错误
- 提供详细的错误信息和解决建议
- 确保工作流正确结束

## 日志和输出

- 仿真日志保存在 `arch/log/simulation_YYYYMMDD_HHMMSS.log`
- 每个步骤都有详细的结构化日志输出
- 错误发生时会提供针对性的解决建议

## 与原Makefile的对应关系

| Makefile目标 | Motia工作流 | 说明 |
|-------------|------------|------|
| `make clean` | `target: "clean"` | 清理构建目录 |
| `make verilog` | `target: "verilog"` | 生成Verilog代码 |
| `make $(BIN)` | 自动触发 | Verilator编译 |
| `make sim` | `target: "sim"` | 运行仿真 |
| `make run` | `target: "run"` | 完整构建+仿真 |

## 环境要求

- Git (用于获取项目根目录)
- Mill (Scala构建工具)
- Verilator (Verilog仿真器)
- 必要的C++编译工具链
- 所需的链接库：dramsim, fesvr, readline

## 故障排除

如果遇到错误，请查看：

1. **清理步骤失败**: 检查目录权限和Git仓库状态
2. **Verilog生成失败**: 确认mill安装正确，Scala代码编译通过
3. **Verilator编译失败**: 检查verilator安装和必要库文件
4. **仿真失败**: 确认workload文件存在，检查系统资源

错误处理器会自动提供详细的解决建议。