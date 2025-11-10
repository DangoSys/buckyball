#!/bin/bash

# Gemmini Ball Generator 启动脚本
# 自动生成 4 个 Ball 并编译验证

set -e

WORK_DIR="/home/daiyongyuan/buckyball"
PROMPT_DIR="$WORK_DIR/workflow/steps/demo/prompt"

echo "================================================"
echo "Gemmini Ball Generator"
echo "================================================"
echo ""
echo "任务：自动生成 4 个 Ball（MatMul, Im2col, Transpose, Norm）"
echo "目标：所有代码能够编译成功"
echo ""
echo "开始执行..."
echo ""

# 读取任务描述和 agent 指令
TASK_PROMPT=$(cat "$PROMPT_DIR/gemmini_task.md")
AGENT_PROMPT=$(cat "$PROMPT_DIR/gemmini_ball_generator.md")

# 调用 Python 脚本启动 agent
cd "$WORK_DIR/workflow/steps/demo"
python3 << 'PYTHON_SCRIPT'
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from steps.demo.simple_gemmini_agent import run_gemmini_generator

# 运行生成器
run_gemmini_generator()
PYTHON_SCRIPT

echo ""
echo "================================================"
echo "任务完成！"
echo "================================================"

