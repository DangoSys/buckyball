# UVM 工作流

BuckyBall 框架中的 UVM (Universal Verification Methodology) 工作流，用于构建和运行 UVM 验证环境。

## API 使用说明

### `builddut`
**端点**: `POST /uvm/builddut`

**功能**: 构建 DUT (Design Under Test)

**参数**:
- **`jobs`** - 并行构建任务数，默认 16

**示例**:
```bash
# 使用默认并行数构建 DUT
bbdev uvm --builddut

# 指定并行任务数
bbdev uvm --builddut "--jobs 8"
```

### `build`
**端点**: `POST /uvm/build`

**功能**: 构建 UVM 可执行文件

**参数**:
- **`jobs`** - 并行构建任务数，默认 16

**示例**:
```bash
# 使用默认并行数构建 UVM
bbdev uvm --build

# 指定并行任务数
bbdev uvm --build "--jobs 8"
```

## 典型工作流程

```bash
# 1. 构建 DUT
bbdev uvm --builddut

# 2. 构建 UVM 环境
bbdev uvm --build
```

**响应格式**:
```json
{
  "status": 200,
  "body": {
    "success": true,
    "processing": false,
    "return_code": 0
  }
}
```
