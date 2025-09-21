# Doc-Agent 工作流

BuckyBall 框架中的文档生成工作流，提供自动化的代码文档生成功能。

## API 使用说明

### `generate`
**端点**: `POST /doc/generate`

**功能**: 为指定目录生成文档

**参数**:
- **`target_path`** [必选] - 目标目录路径
- **`mode`** [必选] - 生成模式，可选值: `"create"`, `"update"`

**示例**:
```bash
# 为指定目录创建新文档
bbdev doc --generate "--target_path arch/src/main/scala/framework --mode create"

# 更新现有文档
bbdev doc --generate "--target_path arch/src/main/scala/framework --mode update"
```

**响应**:
```json
{
  "traceId": "unique-trace-id",
  "status": "success",
  "message": "Documentation generated successfully"
}
```

## 支持的文档类型

- RTL 硬件文档
- 测试文档
- 脚本文档
- 仿真器文档
- 工作流文档

## 注意事项

- 需要配置 AI 模型的 API 密钥
- 生成的文档会自动集成到 mdBook 系统中
- 支持符号链接管理和 SUMMARY.md 自动更新
