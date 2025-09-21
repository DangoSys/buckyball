# Workload 工作流

BuckyBall 框架中的工作负载构建工作流，用于构建测试工作负载和基准程序。

## API 使用说明

### `build`
**端点**: `POST /workload/build`

**功能**: 构建工作负载

**参数**:
- **`workload`** - 指定要构建的工作负载名称

**示例**:
```bash
# 构建指定工作负载
bbdev workload --build "--workload test_program"

# 构建所有工作负载
bbdev workload --build
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

## 注意事项

- 工作负载源码位于 `bb-tests/workload` 目录
- 构建结果通常输出到 `bb-tests/workloads/build` 目录
