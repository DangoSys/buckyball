#!/bin/bash

# Code Agent 测试脚本
# 演示如何使用 Function Calling 功能

BASE_URL="http://localhost:3001"
WORK_DIR="/home/mio/Code/buckyball/workflow/steps/agent"
MODEL="claude-sonnet-4-20250514"

echo "=========================================="
echo "Code Agent Function Calling 测试"
echo "=========================================="
echo ""
echo "⚠️  注意：Function Calling 模式下，AI 会多次调用工具"
echo "    每次请求可能需要 10-30 秒，请耐心等待"
echo ""

echo ""
echo "测试 1: 单次代码生成"
echo "----------------------------"
echo "⏳ 请等待，AI 正在思考和调用工具..."
curl -s -X POST "$BASE_URL/agent/code" \
  -H "Content-Type: application/json" \
  -d "{
    \"promptPath\": \"example_prompt.md\",
    \"model\": \"$MODEL\",
    \"workDir\": \"$WORK_DIR\"
  }" | jq

echo ""
echo ""
echo "测试 2: 代码重构（需要先读取现有文件）"
echo "----------------------------"
echo "⏳ AI 会先读取文件，然后生成代码..."
curl -s -X POST "$BASE_URL/agent/code" \
  -H "Content-Type: application/json" \
  -d "{
    \"promptPath\": \"example_refactor_prompt.md\",
    \"model\": \"$MODEL\",
    \"workDir\": \"$WORK_DIR\"
  }" | jq

echo ""
echo ""
echo "测试 3: 多轮对话 - 第一轮"
echo "----------------------------"
echo "⏳ 创建新会话..."
SESSION_ID="test-session-$(date +%s)"
curl -s -X POST "$BASE_URL/agent/code" \
  -H "Content-Type: application/json" \
  -d "{
    \"promptPath\": \"example_prompt.md\",
    \"workDir\": \"$WORK_DIR\",
    \"model\": \"$MODEL\",
    \"sessionId\": \"$SESSION_ID\"
  }" | jq

echo ""
echo "等待 2 秒..."
sleep 2

echo ""
echo "测试 3: 多轮对话 - 第二轮（使用同一个 session）"
echo "----------------------------"
echo "⏳ 继续之前的会话..."
# 创建一个临时 prompt
cat > /tmp/continue_prompt.md << EOF
# 继续任务

基于刚才创建的代码，请添加一个测试文件 test_calculator.py，包含基本的单元测试。
EOF

curl -s -X POST "$BASE_URL/agent/code" \
  -H "Content-Type: application/json" \
  -d "{
    \"promptPath\": \"/tmp/continue_prompt.md\",
    \"workDir\": \"$WORK_DIR\",
    \"model\": \"$MODEL\",
    \"sessionId\": \"$SESSION_ID\"
  }" | jq

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "查看生成的文件："
echo "  ls -la $WORK_DIR/*.py"
echo ""
echo "检查日志了解 AI 的工具调用过程"
