# Marshal 工作流

BuckyBall 框架中的 Marshal 工作流，用于构建和启动 Marshal 组件。

## API 使用说明

### `build`
**端点**: `POST /marshal/build`

**功能**: 构建 Marshal 组件

**参数**: 无特定参数

**示例**:
```bash
bbdev marshal --build
```

### `launch`
**端点**: `POST /marshal/launch`

**功能**: 启动 Marshal 服务

**参数**: 无特定参数

**示例**:
```bash
bbdev marshal --launch
```

## 典型工作流程

```bash
# 1. 构建 Marshal
bbdev marshal --build

# 2. 启动 Marshal 服务
bbdev marshal --launch
```

## 响应格式

所有 API 调用返回统一格式：
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
