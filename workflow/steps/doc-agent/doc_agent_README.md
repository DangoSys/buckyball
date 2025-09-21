# Doc-Agent 使用文档

## 概述

Doc-Agent 是基于 Motia 框架构建的自动化文档生成系统，能够为代码库中的不同类型目录自动生成高质量的中文技术文档。系统支持多种文档类型，包括 RTL 硬件文档、测试文档、脚本文档、仿真器文档和工作流文档。

## 系统架构

Doc-Agent 采用事件驱动的微服务架构：

```
API 接口 → 事件处理 → 文档生成 → 集成管理 → mdBook 集成
```

- **API Step**: 接收 HTTP 请求，触发文档生成事件
- **Event Step**: 处理文档生成逻辑，调用 LLM API
- **Integration Step**: 管理符号链接和 SUMMARY.md 更新
- **Template System**: 提供多种文档类型的专用模板

## API 接口说明

### 端点信息

- **URL**: `POST /doc/generate`
- **Content-Type**: `application/json`
- **描述**: 生成指定目录的文档

### 请求参数

| 参数名 | 类型 | 必需 | 描述 |
|--------|------|------|------|
| `target_path` | string | 是 | 目标代码目录的相对路径 |
| `mode` | string | 是 | 生成模式：`create` 或 `update` |

#### 模式说明

- **create**: 从零创建新文档，适用于没有现有文档的目录
- **update**: 更新现有文档，保留准确内容，修正过时信息

### 响应格式

#### 成功响应 (200)
```json
{
  "status": "success",
  "message": "文档生成任务已启动",
  "data": {
    "target_path": "arch/src/main/scala/framework",
    "mode": "create",
    "doc_type": "rtl",
    "trace_id": "doc-gen-20241201-001"
  }
}
```

#### 错误响应 (400/500)
```json
{
  "status": "error",
  "message": "错误描述",
  "error_code": "INVALID_PATH",
  "details": {
    "target_path": "提供的路径不存在或无法访问"
  }
}
```

## 使用示例

### 基本用法

#### 1. 生成 RTL 硬件文档
```bash
curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "arch/src/main/scala/framework/builtin",
    "mode": "create"
  }'
```

#### 2. 更新测试文档
```bash
curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "bb-tests/workloads/src",
    "mode": "update"
  }'
```

#### 3. 生成脚本文档
```bash
curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "scripts/docker",
    "mode": "create"
  }'
```

### 批量处理示例

#### 处理整个测试目录
```bash
# 处理所有 bb-tests 子目录
for dir in workloads customext sardine uvbb; do
  curl -X POST http://localhost:8080/doc/generate \
    -H "Content-Type: application/json" \
    -d "{\"target_path\": \"bb-tests/$dir\", \"mode\": \"create\"}"
  sleep 2  # 避免并发过载
done
```

#### 批量更新现有文档
```bash
# 更新所有主要目录的文档
targets=("arch/src/main/scala" "bb-tests/workloads" "scripts" "sims/func-sim" "workflow/steps")

for target in "${targets[@]}"; do
  echo "更新文档: $target"
  curl -X POST http://localhost:8080/doc/generate \
    -H "Content-Type: application/json" \
    -d "{\"target_path\": \"$target\", \"mode\": \"update\"}"
  echo "等待处理完成..."
  sleep 5
done
```

## 支持的文档类型

系统根据目录路径自动识别文档类型：

| 路径模式 | 文档类型 | 模板文件 | 描述 |
|----------|----------|----------|------|
| `arch/src/main/scala/**` | RTL | rtl-doc.md | RTL 硬件模块文档 |
| `bb-tests/workloads/**` | Workloads | workloads-doc.md | 工作负载测试文档 |
| `bb-tests/customext/**` | CustomExt | customext-doc.md | 自定义扩展测试文档 |
| `bb-tests/sardine/**` | Sardine | sardine-doc.md | Sardine 测试框架文档 |
| `bb-tests/uvbb/**` | UVBB | uvbb-doc.md | UVBB 测试文档 |
| `scripts/**` | Script | script-doc.md | 脚本和工具文档 |
| `sims/**` | Simulator | sim-doc.md | 仿真器文档 |
| `workflow/**` | Workflow | workflow-doc.md | 工作流和自动化文档 |

## 文档标准

所有生成的文档遵循统一标准：

### 语言规范
- **主要语言**: 中文
- **技术术语**: 保持英文原文
- **代码注释**: 提供中文解释
- **专业语调**: 避免使用 emoji 和非正式表达

### 格式规范
- **Markdown 格式**: 标准 Markdown 语法
- **代码块**: 使用语法高亮
- **链接**: 使用相对路径
- **图表**: 支持 Mermaid 图表

### 结构规范
不同文档类型有不同的结构要求，但都包含：
- 概述部分
- 代码结构分析
- 详细说明
- 使用示例（如适用）

## 集成功能

### 自动集成到 mdBook

生成的文档会自动集成到项目的 mdBook 文档系统：

1. **符号链接创建**: 在 `docs/bb-note/src/` 下创建对应的目录结构
2. **SUMMARY.md 更新**: 自动添加新文档到目录索引
3. **结构验证**: 确保代码目录和文档目录一一对应

### 目录映射示例

```
代码目录                    →  文档目录
arch/src/main/scala/       →  docs/bb-note/src/arch/src/main/scala/
bb-tests/workloads/        →  docs/bb-note/src/bb-tests/workloads/
scripts/docker/            →  docs/bb-note/src/scripts/docker/
```

## 常见问题和故障排除

### Q1: 文档生成失败，返回 "路径不存在" 错误

**原因**: 提供的 `target_path` 不存在或无法访问

**解决方案**:
```bash
# 检查路径是否存在
ls -la arch/src/main/scala/framework

# 确保使用相对路径，不要以 / 开头
# 正确: "arch/src/main/scala/framework"
# 错误: "/arch/src/main/scala/framework"
```

### Q2: 生成的文档质量不佳或内容不准确

**原因**:
- 目录中代码文件较少或注释不足
- 选择了错误的生成模式
- LLM API 响应异常

**解决方案**:
```bash
# 1. 检查目录内容
find arch/src/main/scala/framework -name "*.scala" | head -10

# 2. 尝试 update 模式而不是 create 模式
curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "arch/src/main/scala/framework",
    "mode": "update"
  }'

# 3. 检查系统日志
tail -f logs/doc-agent.log
```

### Q3: SUMMARY.md 更新失败

**原因**:
- SUMMARY.md 文件权限问题
- 文件格式不符合预期
- 并发更新冲突

**解决方案**:
```bash
# 检查文件权限
ls -la docs/bb-note/src/SUMMARY.md

# 备份并重置 SUMMARY.md
cp docs/bb-note/src/SUMMARY.md docs/bb-note/src/SUMMARY.md.backup

# 检查文件格式
head -20 docs/bb-note/src/SUMMARY.md
```

### Q4: 符号链接创建失败

**原因**:
- 目标目录权限不足
- 磁盘空间不足
- 文件系统不支持符号链接

**解决方案**:
```bash
# 检查磁盘空间
df -h docs/

# 检查权限
ls -la docs/bb-note/src/

# 手动测试符号链接创建
ln -s ../../../arch/src/main/scala/framework docs/bb-note/src/arch/src/main/scala/framework
```

### Q5: API 请求超时

**原因**:
- LLM API 响应慢
- 目录文件过多，分析时间长
- 网络连接问题

**解决方案**:
```bash
# 增加请求超时时间
curl -X POST http://localhost:8080/doc/generate \
  --max-time 300 \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "arch/src/main/scala/framework",
    "mode": "create"
  }'

# 分批处理大目录
# 不要直接处理 arch/src/main/scala，而是处理其子目录
```

## 性能优化建议

### 1. 批量处理优化
```bash
# 使用并行处理（谨慎使用，避免 API 限制）
parallel -j 2 curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{\"target_path\": \"{}\", \"mode\": \"create\"}' \
  ::: arch/src/main/scala/framework arch/src/main/scala/builtin
```

### 2. 增量更新策略
```bash
# 只更新最近修改的目录
find arch/src/main/scala -type d -mtime -7 | while read dir; do
  if [[ -f "$dir/README.md" ]]; then
    curl -X POST http://localhost:8080/doc/generate \
      -H "Content-Type: application/json" \
      -d "{\"target_path\": \"$dir\", \"mode\": \"update\"}"
  fi
done
```

### 3. 监控和日志
```bash
# 监控 API 响应时间
time curl -X POST http://localhost:8080/doc/generate \
  -H "Content-Type: application/json" \
  -d '{
    "target_path": "arch/src/main/scala/framework",
    "mode": "create"
  }'

# 查看详细日志
tail -f logs/doc-agent.log | grep -E "(ERROR|WARN|生成完成)"
```

## 配置说明

### 环境变量

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `DOC_AGENT_PORT` | API 服务端口 | 8080 |
| `LLM_API_KEY` | LLM API 密钥 | 必需设置 |
| `LLM_API_URL` | LLM API 端点 | 必需设置 |
| `DOC_OUTPUT_BASE` | 文档输出基础路径 | `docs/bb-note/src` |
| `TEMPLATE_BASE_PATH` | 模板文件基础路径 | `workflow/prompts/doc` |

### 配置文件示例

创建 `.env` 文件：
```bash
# LLM API 配置
LLM_API_KEY=your_api_key_here
LLM_API_URL=https://api.openai.com/v1/chat/completions

# 文档系统配置
DOC_OUTPUT_BASE=docs/bb-note/src
TEMPLATE_BASE_PATH=workflow/prompts/doc

# 性能配置
MAX_CONCURRENT_REQUESTS=3
REQUEST_TIMEOUT=300
```

## 开发和调试

### 本地开发环境设置

```bash
# 1. 安装依赖
cd workflow
npm install

# 2. 启动 Motia 服务
npm run dev

# 3. 测试 API 连接
curl http://localhost:8080/health
```

### 调试模式

```bash
# 启用详细日志
export DEBUG=doc-agent:*
npm run dev

# 测试单个组件
node -e "
const { loadTemplate } = require('./steps/doc-agent/template_loader');
console.log(loadTemplate('rtl', 'arch/src/main/scala/test'));
"
```

## 版本信息

- **当前版本**: 1.0.0
- **Motia 框架版本**: 兼容 v2.x
- **支持的 Node.js 版本**: >= 16.0.0
- **最后更新**: 2024年12月

## 支持和反馈

如遇到问题或需要功能改进，请：

1. 检查本文档的故障排除部分
2. 查看系统日志文件
3. 在项目仓库中创建 Issue
4. 联系开发团队

---

*本文档随系统更新而持续维护，请定期查看最新版本。*
