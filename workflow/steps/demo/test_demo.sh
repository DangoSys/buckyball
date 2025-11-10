#!/bin/bash
# Gemmini Ball Generator - 简化版启动脚本

set -e

WORK_DIR="/home/daiyongyuan/buckyball"
MODEL="qwen3-235b-a22b-instruct-2507"

echo "============================================================"
echo "Gemmini Ball Generator - 简化版"
echo "============================================================"
echo ""
echo "任务：自动生成 4 个 Ball（MatMul, Im2col, Transpose, Norm）"
echo "目标：所有代码能够编译成功"
echo ""

# 检查是否有参数
if [ "$1" == "api" ]; then
  echo "⚠️  警告：API 模式需要 bbdev 服务运行"
  echo "使用 API 模式..."
  BBDEV_URL="http://localhost:3001"
  SESSION_ID="gemmini-$(date +%s)"
  
  echo "Session ID: $SESSION_ID"
  echo "BBDEV URL: $BBDEV_URL"
  echo ""
  
  # 检查 bbdev 是否运行
  if ! curl -s "$BBDEV_URL/health" > /dev/null 2>&1; then
    echo "❌ 错误：bbdev 服务未运行"
    echo "请先启动 bbdev 服务或使用直接模式（不带参数）"
    exit 1
  fi
  
  curl -X POST "$BBDEV_URL/agent" \
    -H "Content-Type: application/json" \
    -d "{
      \"agentRole\": \"gemmini\",
      \"promptPath\": \"workflow/steps/demo/prompt/gemmini_task.md\",
      \"systemPromptPath\": \"workflow/steps/demo/prompt/gemmini_ball_generator.md\",
      \"workDir\": \"$WORK_DIR\",
      \"model\": \"$MODEL\",
      \"sessionId\": \"$SESSION_ID\"
    }"
  
  echo ""
  echo "✅ Agent 已启动"
else
  echo "使用直接运行模式（推荐）..."
  echo ""
  
  # 检查 Python 依赖
  python3 -c "import httpx" 2>/dev/null || {
    echo "⚠️  警告：缺少 httpx 依赖"
    echo "安装：pip3 install httpx python-dotenv"
    echo ""
  }
  
  cd "$WORK_DIR/workflow/steps/demo"
  python3 simple_gemmini_agent.py
fi

echo ""
echo "============================================================"
