# Sardine 工作流

BuckyBall 框架中的 Sardine 工作流，用于运行 Sardine 相关任务。

## API 使用说明

### `run`
**端点**: `POST /sardine/run`

**功能**: 运行 Sardine 任务

**参数**:
- **`workload`** - 指定要运行的工作负载

**示例**:
```bash
# 运行指定工作负载
bbdev sardine --run "--workload /path/to/workload"

# 运行默认工作负载
bbdev sardine --run
```

**响应**:
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
