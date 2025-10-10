# BuckyBall MCP Server

将 bbdev 工具封装为 MCP 服务，可以在 Claude/Cursor 中直接使用。

## 安装

```bash
pip install mcp
```

## 配置

Cursor

在 MCP 配置中添加：

```json
{
  "mcpServers": {
    "bbdev": {
      "command": "bash",
      "args": ["-c", "cd /path/to/your/buckyball && source env.sh && python /path/to/your/buckyball/workflow/mcp-server/server.py"]
    }
  }
}
```

## 使用

配置后重启客户端，就可以自然语言调用 bbdev 所有功能了。

示例：
- "用 verilator 运行 gelu_test"
- "清理构建目录"
- "运行 sardine 测试"
