#!/usr/bin/env python3
"""
Gemmini Ball Generator - 简化版单一 Agent
自动生成 4 个 Ball（MatMul, Im2col, Transpose, Norm）并编译验证
"""

import os
import sys
import json
import httpx
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 添加 utils 路径
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
  sys.path.insert(0, utils_path)

from utils.stream_run import stream_run_logger

# 配置
WORK_DIR = Path("/home/daiyongyuan/buckyball")
PROMPT_DIR = WORK_DIR / "workflow/steps/demo/prompt"
BUILD_SCRIPT = WORK_DIR / "scripts/build_gemmini.sh"
BUILD_LOG = WORK_DIR / "build_logs/gemmini_build.log"

# LLM API 配置
# 支持多种环境变量名，兼容旧系统
API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("BASE_URL") or os.getenv("LLM_BASE_URL") or "http://localhost:8000/v1"
API_KEY = os.getenv("API_KEY") or os.getenv("LLM_API_KEY") or "dummy-key"
MODEL = os.getenv("MODEL") or "qwen3-235b-a22b-instruct-2507"

# Agent 工具定义
TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "读取文件内容",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "文件路径"}
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "write_file",
      "description": "写入文件内容",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "文件路径"},
          "content": {"type": "string", "description": "文件内容"}
        },
        "required": ["path", "content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_files",
      "description": "列出目录下的文件",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "目录路径"}
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "make_dir",
      "description": "创建目录",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "目录路径"}
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "run_build",
      "description": "运行编译脚本并返回结果",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "grep_files",
      "description": "在文件中搜索内容",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "搜索路径"},
          "pattern": {"type": "string", "description": "搜索模式"}
        },
        "required": ["path", "pattern"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "delete_file",
      "description": "删除文件",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "要删除的文件路径"}
        },
        "required": ["path"]
      }
    }
  }
]


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
  """执行工具调用"""
  try:
    if tool_name == "read_file":
      path = arguments["path"]
      full_path = WORK_DIR / path if not path.startswith("/") else Path(path)
      if full_path.exists():
        return full_path.read_text()
      else:
        return f"Error: File not found: {path}"
    
    elif tool_name == "write_file":
      path = arguments["path"]
      content = arguments["content"]
      full_path = WORK_DIR / path if not path.startswith("/") else Path(path)
      full_path.parent.mkdir(parents=True, exist_ok=True)
      full_path.write_text(content)
      return f"Success: File written to {path}"
    
    elif tool_name == "list_files":
      path = arguments["path"]
      full_path = WORK_DIR / path if not path.startswith("/") else Path(path)
      if full_path.exists() and full_path.is_dir():
        files = [str(f.relative_to(full_path)) for f in full_path.iterdir()]
        return "\n".join(files)
      else:
        return f"Error: Directory not found: {path}"
    
    elif tool_name == "make_dir":
      path = arguments["path"]
      full_path = WORK_DIR / path if not path.startswith("/") else Path(path)
      full_path.mkdir(parents=True, exist_ok=True)
      return f"Success: Directory created: {path}"
    
    elif tool_name == "run_build":
      # 运行编译脚本
      result = subprocess.run(
        ["bash", str(BUILD_SCRIPT), "build"],
        capture_output=True,
        text=True,
        timeout=600
      )
      
      # 读取日志
      if BUILD_LOG.exists():
        log_content = BUILD_LOG.read_text()
        if "Compilation completed successfully" in log_content:
          return json.dumps({
            "status": "success",
            "message": "编译成功",
            "log_tail": log_content[-1000:]  # 返回最后1000字符
          })
        else:
          # 提取错误信息
          error_lines = [line for line in log_content.split("\n") if "[error]" in line]
          return json.dumps({
            "status": "failed",
            "message": "编译失败",
            "errors": error_lines[:20],  # 返回前20个错误
            "log_tail": log_content[-2000:]  # 返回最后2000字符
          })
      else:
        return json.dumps({"status": "failed", "message": "日志文件不存在"})
    
    elif tool_name == "grep_files":
      path = arguments["path"]
      pattern = arguments["pattern"]
      full_path = WORK_DIR / path if not path.startswith("/") else Path(path)
      
      # 使用 grep 搜索
      result = subprocess.run(
        ["grep", "-r", pattern, str(full_path)],
        capture_output=True,
        text=True
      )
      return result.stdout if result.returncode == 0 else f"No matches found for: {pattern}"
    
    elif tool_name == "delete_file":
      path = arguments["path"]
      full_path = WORK_DIR / path if not path.startswith("/") else Path(path)
      if full_path.exists():
        full_path.unlink()
        return f"Success: File deleted: {path}"
      else:
        return f"Error: File not found: {path}"
    
    else:
      return f"Error: Unknown tool: {tool_name}"
  
  except Exception as e:
    return f"Error executing {tool_name}: {str(e)}"


def run_gemmini_generator():
  """运行 Gemmini Ball Generator"""
  
  print("\n" + "="*60)
  print("Gemmini Ball Generator - 自动生成 4 个 Ball")
  print("="*60 + "\n")
  
  # 显示配置信息
  print("📋 配置信息:")
  print(f"  API_BASE_URL: {API_BASE_URL}")
  print(f"  MODEL: {MODEL}")
  print(f"  API_KEY: {API_KEY[:20]}..." if len(API_KEY) > 20 else f"  API_KEY: {API_KEY}")
  print("")
  
  # 读取 prompt
  task_prompt = (PROMPT_DIR / "gemmini_task.md").read_text()
  agent_prompt = (PROMPT_DIR / "gemmini_ball_generator.md").read_text()
  
  # 初始化消息
  messages = [
    {
      "role": "system",
      "content": agent_prompt
    },
    {
      "role": "user",
      "content": f"{task_prompt}\n\n**立即开始为 matmul Ball 生成代码！**"
    }
  ]
  
  # Agent 循环
  max_iterations = 200  # 最多200轮对话（增加以处理复杂的错误修复）
  iteration = 0
  balls_completed = []
  
  with httpx.Client(timeout=600.0) as client:
    while iteration < max_iterations:
      iteration += 1
      print(f"\n[迭代 {iteration}]")
      
      # 调用 LLM
      try:
        response = client.post(
          f"{API_BASE_URL}/chat/completions",
          json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0.7,
            "max_tokens": 4000
          },
          headers={"Authorization": f"Bearer {API_KEY}"}
        )
        response.raise_for_status()
        result = response.json()
        
      except Exception as e:
        print(f"❌ API 调用失败: {e}")
        break
      
      # 解析响应
      choice = result["choices"][0]
      message = choice["message"]
      messages.append(message)
      
      # 检查是否有工具调用
      if choice.get("finish_reason") == "tool_calls" and message.get("tool_calls"):
        print(f"🔧 执行 {len(message['tool_calls'])} 个工具调用")
        
        # 执行所有工具调用
        for tool_call in message["tool_calls"]:
          func_name = tool_call["function"]["name"]
          func_args = json.loads(tool_call["function"]["arguments"])
          
          print(f"  - {func_name}({json.dumps(func_args, ensure_ascii=False)[:80]}...)")
          
          # 执行工具
          result_str = execute_tool(func_name, func_args)
          
          # 添加工具结果到消息
          messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": result_str
          })
          
          # 检查是否是编译结果
          if func_name == "run_build":
            try:
              build_result = json.loads(result_str)
              if build_result["status"] == "success":
                print(f"    ✅ 编译成功")
              else:
                print(f"    ❌ 编译失败，需要修复")
            except:
              pass
      
      # 检查是否完成
      elif choice.get("finish_reason") == "stop":
        content = message.get("content", "")
        print(f"💬 Agent: {content[:200]}...")
        
        # 检查是否提到完成了某个 Ball
        for ball in ["matmul", "im2col", "transpose", "norm"]:
          if ball not in balls_completed and (
            f"{ball}" in content.lower() and "成功" in content
          ):
            balls_completed.append(ball)
            print(f"✅ {ball.upper()} Ball 完成！")
        
        # 检查是否所有 Ball 都完成
        if len(balls_completed) >= 4:
          print("\n" + "="*60)
          print("🎉 所有 4 个 Ball 生成完成！")
          print("="*60 + "\n")
          break
        
        # 否则，继续下一个 Ball
        # Agent 会自动继续，不需要额外输入
        
      else:
        print(f"⚠️  未知的完成原因: {choice.get('finish_reason')}")
        break
      
      # 防止无限循环
      if iteration >= max_iterations:
        print(f"\n⚠️  达到最大迭代次数 {max_iterations}，停止执行")
        break
  
  # 总结
  print("\n" + "="*60)
  print("执行总结")
  print("="*60)
  print(f"总迭代次数: {iteration}")
  print(f"完成的 Ball: {', '.join(balls_completed) if balls_completed else '无'}")
  print("")
  
  if len(balls_completed) >= 4:
    print("✅ 任务成功完成！")
    return 0
  else:
    print("❌ 任务未完全完成")
    return 1


if __name__ == "__main__":
  sys.exit(run_gemmini_generator())

