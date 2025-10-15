#!/bin/bash
# Start Gemmini NPU Agent Demo

set -e


BBDEV_URL="http://localhost:3001"
WORK_DIR="/home/mio/Code/buckyball"
# MODEL="claude-sonnet-4-20250514"
# MODEL="gpt-4o-mini"
MODEL="qwen3-235b-a22b-instruct-2507"


MAIN_PROMPT_PATH="workflow/steps/demo/prompt/task/gemmini_npu.md"

# 生成唯一的 session ID（用于 Redis 存储）
SESSION_ID="master-$(date +%s)"

echo "Starting master agent..."
echo "Session ID: $SESSION_ID"
echo ""

curl -X POST "$BBDEV_URL/agent" \
  -H "Content-Type: application/json" \
  -d "{
    \"agentRole\": \"master\",
    \"promptPath\": \"$MAIN_PROMPT_PATH\",
    \"workDir\": \"$WORK_DIR\",
    \"model\": \"$MODEL\",
    \"sessionId\": \"$SESSION_ID\"
  }"

echo ""
echo "Worker agent started"
