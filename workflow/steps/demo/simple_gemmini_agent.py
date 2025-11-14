#!/usr/bin/env python3
"""
Gemmini Ball Generator - 自动化多任务执行器
自动顺序执行：
  任务1: 生成 4 个基础 Ball（MatMul, Im2col, Transpose, Norm）
  任务2: 设计 ABFT 可靠性脉动阵列（WS/OS + ABFT）
  任务3: 设计可配置位宽脉动阵列（WS/OS + Quantization）
  任务4: 设计三数据流脉动阵列（WS/OS/RS）
"""

import os
import sys
import json
import httpx
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime

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
API_BASE_URL = (
    os.getenv("API_BASE_URL")
    or os.getenv("BASE_URL")
    or os.getenv("LLM_BASE_URL")
    or "http://localhost:8000/v1"
)
API_KEY = os.getenv("API_KEY") or os.getenv("LLM_API_KEY") or "dummy-key"
MODEL = os.getenv("MODEL") or "qwen3-235b-a22b-instruct-2507"

# ============================================================================
# 🎯 快速切换任务 - 只需修改下面这个数字！
# ============================================================================
TASK_TO_RUN = 4  # 改成 1, 2, 3, 或 4 即可切换任务
TOTAL_RUNS = 20  # 连续运行次数（无论成功失败都继续）
# ============================================================================

# 所有可用任务定义
ALL_TASKS = {
    1: {
        "id": 1,
        "name": "基础 Ball 生成",
        "desc": "生成 MatMul, Im2col, Transpose, Norm 四个基础 Ball",
        "task_file": "task/gemmini_task.md",
        "user_prompt": "**立即开始生成 4 个 Gemmini Ball！**",
        "success_keywords": ["matmul", "im2col", "transpose", "norm"],
        "max_iterations": 100,
    },
    2: {
        "id": 2,
        "name": "ABFT 可靠性脉动阵列",
        "desc": "设计支持 WS/OS 数据流和 ABFT 可靠性机制的脉动阵列",
        "task_file": "task/task2_abft_systolic.md",
        "user_prompt": "**立即开始设计 ABFT 可靠性脉动阵列！**",
        "success_keywords": ["abft"],
        "max_iterations": 150,
    },
    3: {
        "id": 3,
        "name": "可配置位宽脉动阵列",
        "desc": "设计支持 WS/OS 和可配置数据位宽/量化精度的脉动阵列",
        "task_file": "task/task3_configurable_systolic.md",
        "user_prompt": "**立即开始设计可配置位宽脉动阵列！**",
        "success_keywords": ["configurable", "quantization"],
        "max_iterations": 120,
    },
    4: {
        "id": 4,
        "name": "三数据流脉动阵列",
        "desc": "设计支持 WS/OS/RS 三种数据流的脉动阵列",
        "task_file": "task/task4_triple_dataflow_systolic.md",
        "user_prompt": "**立即开始设计三数据流脉动阵列！**",
        "success_keywords": ["ws", "os", "rs"],
        "max_iterations": 150,
    },
}

# 根据配置选择任务
if TASK_TO_RUN not in ALL_TASKS:
    print(f"❌ 错误：任务 {TASK_TO_RUN} 不存在！请选择 1, 2, 3, 或 4")
    sys.exit(1)

TASKS = [ALL_TASKS[TASK_TO_RUN]]

# Agent 工具定义
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "文件路径"}},
                "required": ["path"],
            },
        },
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
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "列出目录下的文件",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "目录路径"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_dir",
            "description": "创建目录",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "目录路径"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_build",
            "description": "运行编译脚本并返回结果",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
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
                    "pattern": {"type": "string", "description": "搜索模式"},
                },
                "required": ["path", "pattern"],
            },
        },
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
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_test",
            "description": "编译并运行 C 测试文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "test_file": {
                        "type": "string",
                        "description": "C 测试文件路径（如 tests/gemmini_abft_test.c）",
                    }
                },
                "required": ["test_file"],
            },
        },
    },
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
                timeout=600,
            )

            # 读取日志
            if BUILD_LOG.exists():
                log_content = BUILD_LOG.read_text()
                if "Compilation completed successfully" in log_content:
                    return json.dumps(
                        {
                            "status": "success",
                            "message": "编译成功",
                            "log_tail": log_content[-1000:],  # 返回最后1000字符
                        }
                    )
                else:
                    # 提取错误信息
                    error_lines = [
                        line for line in log_content.split("\n") if "[error]" in line
                    ]
                    return json.dumps(
                        {
                            "status": "failed",
                            "message": "编译失败",
                            "errors": error_lines[:20],  # 返回前20个错误
                            "log_tail": log_content[-2000:],  # 返回最后2000字符
                        }
                    )
            else:
                return json.dumps({"status": "failed", "message": "日志文件不存在"})

        elif tool_name == "grep_files":
            path = arguments["path"]
            pattern = arguments["pattern"]
            full_path = WORK_DIR / path if not path.startswith("/") else Path(path)

            # 使用 grep 搜索
            result = subprocess.run(
                ["grep", "-r", pattern, str(full_path)], capture_output=True, text=True
            )
            return (
                result.stdout
                if result.returncode == 0
                else f"No matches found for: {pattern}"
            )

        elif tool_name == "delete_file":
            path = arguments["path"]
            full_path = WORK_DIR / path if not path.startswith("/") else Path(path)
            if full_path.exists():
                full_path.unlink()
                return f"Success: File deleted: {path}"
            else:
                return f"Error: File not found: {path}"

        elif tool_name == "run_test":
            test_file = arguments["test_file"]
            test_path = (
                WORK_DIR / test_file
                if not test_file.startswith("/")
                else Path(test_file)
            )

            if not test_path.exists():
                return json.dumps(
                    {"status": "error", "message": f"测试文件不存在: {test_file}"}
                )

            # 编译测试文件
            output_binary = test_path.with_suffix("")
            compile_cmd = [
                "gcc",
                "-o",
                str(output_binary),
                str(test_path),
                "-I/home/daiyongyuan/buckyball/arch/src/main/c",
                "-lm",
                "-Wall",
            ]

            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(WORK_DIR),
                )

                if compile_result.returncode != 0:
                    return json.dumps(
                        {
                            "status": "compile_failed",
                            "message": "C 测试编译失败",
                            "stdout": compile_result.stdout,
                            "stderr": compile_result.stderr,
                        }
                    )

                # 运行测试
                run_result = subprocess.run(
                    [str(output_binary)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(WORK_DIR),
                )

                # 清理可执行文件
                if output_binary.exists():
                    output_binary.unlink()

                if run_result.returncode == 0:
                    return json.dumps(
                        {
                            "status": "success",
                            "message": "测试通过",
                            "stdout": run_result.stdout,
                            "stderr": run_result.stderr,
                        }
                    )
                else:
                    return json.dumps(
                        {
                            "status": "test_failed",
                            "message": f"测试失败 (退出码: {run_result.returncode})",
                            "stdout": run_result.stdout,
                            "stderr": run_result.stderr,
                        }
                    )

            except subprocess.TimeoutExpired:
                return json.dumps(
                    {"status": "timeout", "message": "测试运行超时（30秒）"}
                )
            except Exception as e:
                return json.dumps(
                    {"status": "error", "message": f"运行测试时出错: {str(e)}"}
                )

        else:
            return f"Error: Unknown tool: {tool_name}"

    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def run_single_task(task_config: Dict[str, Any], agent_prompt: str) -> Dict[str, Any]:
    """运行单个任务，返回统计信息"""

    print("\n" + "=" * 80)
    print(f"🚀 任务 {task_config['id']}: {task_config['name']}")
    print(f"   {task_config['desc']}")
    print("=" * 80 + "\n")

    # 读取任务 prompt
    task_file = PROMPT_DIR / task_config["task_file"]
    if not task_file.exists():
        print(f"❌ 任务文件不存在: {task_file}")
        return {"success": False, "tokens": 0}

    task_prompt = task_file.read_text()

    # 替换 prompt 中的占位符
    build_script_path = os.getenv("BUILD_SCRIPT_PATH") or str(BUILD_SCRIPT)
    build_log_path = os.getenv("BUILD_LOG_PATH") or str(BUILD_LOG)
    task_prompt = task_prompt.replace("{BUILD_SCRIPT_PATH}", build_script_path)
    task_prompt = task_prompt.replace("{BUILD_LOG_PATH}", build_log_path)
    task_prompt = task_prompt.replace("{WORK_DIR}", str(WORK_DIR))

    # 初始化消息
    messages = [
        {"role": "system", "content": agent_prompt},
        {"role": "user", "content": f"{task_prompt}\n\n{task_config['user_prompt']}"},
    ]

    # Agent 循环
    max_iterations = task_config["max_iterations"]
    iteration = 0
    success_count = 0
    last_build_success = False
    last_test_success = False  # 追踪测试是否通过
    consecutive_json_errors = 0  # 连续JSON错误计数

    # Token 统计
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    with httpx.Client(timeout=600.0) as client:
        while iteration < max_iterations:
            iteration += 1
            print(f"\n[任务 {task_config['id']} - 迭代 {iteration}]")
            
            # ⚡ 关键修复：如果测试已通过，立即返回成功（避免无限迭代）
            if last_test_success:
                print(f"\n✅ 任务 {task_config['id']} 完成！（测试已通过）")
                print("📊 Token 使用统计:")
                print(f"   输入 tokens: {total_prompt_tokens:,}")
                print(f"   输出 tokens: {total_completion_tokens:,}")
                print(f"   总计 tokens: {total_tokens:,}")
                return {
                    "success": True,
                    "tokens": total_tokens,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                }

            # 调用 LLM
            try:
                response = client.post(
                    f"{API_BASE_URL}/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "tools": TOOLS,
                        "temperature": 0.7,
                        "max_tokens": 4000,
                    },
                    headers={"Authorization": f"Bearer {API_KEY}"},
                )
                response.raise_for_status()
                result = response.json()

            except Exception as e:
                print(f"❌ API 调用失败: {e}")
                return {"success": False, "tokens": total_tokens}

            # 统计 token 使用
            if "usage" in result:
                usage = result["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                tokens = usage.get("total_tokens", 0)

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += tokens

            # 解析响应
            choice = result["choices"][0]
            message = choice["message"]
            messages.append(message)

            # 检查是否有工具调用
            if choice.get("finish_reason") == "tool_calls" and message.get(
                "tool_calls"
            ):
                print(f"🔧 执行 {len(message['tool_calls'])} 个工具调用")

                # 执行所有工具调用
                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]

                    # 解析工具参数（带错误处理）
                    try:
                        func_args = json.loads(tool_call["function"]["arguments"])
                        consecutive_json_errors = 0  # 重置错误计数
                    except json.JSONDecodeError as e:
                        consecutive_json_errors += 1
                        print(f"  ⚠️  JSON 解析错误 ({consecutive_json_errors}/3): {e}")
                        print(f"     跳过此工具调用: {func_name}")

                        # 根据工具类型给出具体建议
                        if func_name == "write_file":
                            if consecutive_json_errors >= 3:
                                error_msg = """Error: JSON parsing failed 3 times in a row!

CRITICAL: You must change your strategy immediately.

Required actions:
1. Use read_file to check what files already exist in the target directory
2. DON'T regenerate large files - build incrementally
3. Focus on running build to check current compilation status
4. Fix specific errors one at a time

DO NOT try to write large files again. Check the current state first."""
                            else:
                                error_msg = f"""Error: JSON parsing failed - {str(e)}

This usually happens when the file content is too long or contains unescaped special characters.

Solutions:
1. Split into multiple smaller files (e.g., separate PE, Controller, DataPath)
2. Write a minimal skeleton first, then add details in subsequent calls
3. Ensure all strings are properly escaped in JSON

Please try a different approach."""
                        else:
                            error_msg = (
                                f"Error: JSON parsing failed - {str(e)}. "
                                "Please simplify your arguments and try again."
                            )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": error_msg,
                            }
                        )
                        continue

                    print(
                        f"  - {func_name}({json.dumps(func_args, ensure_ascii=False)[:80]}...)"
                    )

                    # 执行工具
                    try:
                        result_str = execute_tool(func_name, func_args)
                    except Exception as e:
                        print(f"    ❌ 工具执行错误: {e}")
                        result_str = f"Error executing tool: {str(e)}"

                    # 添加工具结果到消息
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result_str,
                        }
                    )

                    # 检查是否是编译结果
                    if func_name == "run_build":
                        try:
                            build_result = json.loads(result_str)
                            if build_result["status"] == "success":
                                print("    ✅ 编译成功")
                                last_build_success = True
                                success_count += 1
                            else:
                                print("    ❌ 编译失败，需要修复")
                                last_build_success = False
                        except Exception:
                            pass

                    # 检查是否是测试结果
                    elif func_name == "run_test":
                        try:
                            test_result = json.loads(result_str)
                            if test_result["status"] == "success":
                                print("    ✅ 测试通过")
                                success_count += 1
                                last_test_success = True  # 标记测试通过
                            elif test_result["status"] == "test_failed":
                                print("    ❌ 测试失败，需要修复")
                                print(
                                    f"    输出: {test_result.get('stdout', '')[:200]}"
                                )
                                last_test_success = False
                            elif test_result["status"] == "compile_failed":
                                print("    ❌ C 测试编译失败")
                                print(
                                    f"    错误: {test_result.get('stderr', '')[:200]}"
                                )
                                last_test_success = False
                            elif test_result["status"] == "timeout":
                                print("    ⏱️  测试超时（30秒）")
                                last_test_success = False
                        except Exception:
                            pass

            # 检查是否完成
            elif choice.get("finish_reason") == "stop":
                content = message.get("content", "")
                print(f"💬 Agent: {content[:200]}...")

                # 检查任务是否完成
                # 条件1：测试通过了
                # 条件2：编译成功 + Agent说完成了
                task_complete = False
                
                if last_test_success:
                    print(f"\n✅ 任务 {task_config['id']} 完成！（测试通过）")
                    task_complete = True
                elif last_build_success and any(
                    kw in content.lower() for kw in ["完成", "成功", "successfully", "finished", "done", "completed"]
                ):
                    print(f"\n✅ 任务 {task_config['id']} 完成！（编译成功且Agent确认）")
                    task_complete = True
                
                if task_complete:
                    print("📊 Token 使用统计:")
                    print(f"   输入 tokens: {total_prompt_tokens:,}")
                    print(f"   输出 tokens: {total_completion_tokens:,}")
                    print(f"   总计 tokens: {total_tokens:,}")
                    return {
                        "success": True,
                        "tokens": total_tokens,
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                    }

                # 如果没有编译成功但 Agent 停止了，继续推动
                if not last_build_success:
                    # 添加用户消息推动继续
                    messages.append(
                        {"role": "user", "content": "继续修复编译错误，直到编译成功。"}
                    )

            else:
                print(f"⚠️  未知的完成原因: {choice.get('finish_reason')}")
                break

            # 防止无限循环
            if iteration >= max_iterations:
                print(f"\n⚠️  达到最大迭代次数 {max_iterations}")
                # 如果至少编译成功一次，认为基本完成
                if last_build_success:
                    print(f"✅ 任务 {task_config['id']} 基本完成（最后一次编译成功）")
                    print("📊 Token 使用统计:")
                    print(f"   输入 tokens: {total_prompt_tokens:,}")
                    print(f"   输出 tokens: {total_completion_tokens:,}")
                    print(f"   总计 tokens: {total_tokens:,}")
                    return {
                        "success": True,
                        "tokens": total_tokens,
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                    }
                break

    # 任务失败
    print(f"\n❌ 任务 {task_config['id']} 未完成")
    print("📊 Token 使用统计:")
    print(f"   输入 tokens: {total_prompt_tokens:,}")
    print(f"   输出 tokens: {total_completion_tokens:,}")
    print(f"   总计 tokens: {total_tokens:,}")
    return {
        "success": False,
        "tokens": total_tokens,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }


def run_gemmini_generator():
    """运行 Gemmini Ball Generator - 连续多次执行"""

    total_start_time = datetime.now()

    print("\n" + "=" * 80)
    print("🎯 Gemmini NPU 自动化多任务生成器 - 批量测试模式")
    print("=" * 80)
    print(f"开始时间: {total_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔄 连续运行次数: {TOTAL_RUNS} 次")
    print("")

    # 显示当前运行的任务
    print("🚀 当前任务:")
    for task in TASKS:
        print(f"  任务 {task['id']}: {task['name']}")
        print(f"  描述: {task['desc']}")
        print(f"  最大迭代: {task['max_iterations']}")
    print("")
    print("💡 提示: 要切换任务，请修改文件第 51 行的 TASK_TO_RUN 变量")
    print("💡 提示: 要修改运行次数，请修改文件第 52 行的 TOTAL_RUNS 变量")
    print("")

    # 显示配置信息
    print("📋 配置信息:")
    print(f"  API_BASE_URL: {API_BASE_URL}")
    print(f"  MODEL: {MODEL}")
    print(
        f"  API_KEY: {API_KEY[:20]}..."
        if len(API_KEY) > 20
        else f"  API_KEY: {API_KEY}"
    )
    print("")

    # 读取 Agent prompt（所有任务共用）
    agent_prompt = (PROMPT_DIR / "gemmini_ball_generator.md").read_text()

    # 替换 prompt 中的占位符
    build_script_path = os.getenv("BUILD_SCRIPT_PATH") or str(BUILD_SCRIPT)
    build_log_path = os.getenv("BUILD_LOG_PATH") or str(BUILD_LOG)
    agent_prompt = agent_prompt.replace("{BUILD_SCRIPT_PATH}", build_script_path)
    agent_prompt = agent_prompt.replace("{BUILD_LOG_PATH}", build_log_path)
    agent_prompt = agent_prompt.replace("{WORK_DIR}", str(WORK_DIR))

    # 统计所有运行的结果
    all_runs_results = []
    
    # 🔄 外层循环：连续运行 TOTAL_RUNS 次
    for run_number in range(1, TOTAL_RUNS + 1):
        run_start_time = datetime.now()
        
        print("\n" + "━" * 80)
        print(f"🔄 第 {run_number}/{TOTAL_RUNS} 次运行")
        print("━" * 80)
        
        # 执行所有任务
        results = []
        for task in TASKS:
            task_result = run_single_task(task, agent_prompt)
            results.append(
                {
                    "run": run_number,
                    "task_id": task["id"],
                    "task_name": task["name"],
                    "success": task_result.get("success", False),
                    "tokens": task_result.get("tokens", 0),
                    "prompt_tokens": task_result.get("prompt_tokens", 0),
                    "completion_tokens": task_result.get("completion_tokens", 0),
                }
            )

            # 无论成功失败都继续，不中断
            if not task_result.get("success", False):
                print(f"\n⚠️  任务 {task['id']} 失败，继续下一个任务...")
        
        all_runs_results.extend(results)
        
        run_end_time = datetime.now()
        run_duration = run_end_time - run_start_time
        
        # 每次运行后的小结
        print(f"\n✅ 第 {run_number} 次运行完成，耗时: {run_duration}")
        success_count = sum(1 for r in results if r["success"])
        print(f"   本次成功: {success_count}/{len(results)}")
        
        # 如果还有下一次运行，稍微等待一下
        if run_number < TOTAL_RUNS:
            print(f"   准备第 {run_number + 1} 次运行...\n")

    # 最终总结
    total_end_time = datetime.now()
    total_duration = total_end_time - total_start_time

    print("\n" + "=" * 80)
    print(f"📊 批量测试最终总结 - {TOTAL_RUNS} 次运行")
    print("=" * 80)
    print(f"开始时间: {total_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {total_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_duration}")
    print("")
    
    # 统计每次运行的情况
    print("📋 每次运行结果:")
    for run_num in range(1, TOTAL_RUNS + 1):
        run_results = [r for r in all_runs_results if r["run"] == run_num]
        success_count = sum(1 for r in run_results if r["success"])
        total_tasks = len(run_results)
        status_icon = "✅" if success_count == total_tasks else "❌"
        print(f"  第 {run_num:2d} 次: {status_icon} {success_count}/{total_tasks} 成功")
    print("")
    
    # 总体统计
    total_attempts = len(all_runs_results)
    total_success = sum(1 for r in all_runs_results if r["success"])
    success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
    
    print("📊 总体统计:")
    print(f"   总运行次数: {TOTAL_RUNS} 次")
    print(f"   总任务执行: {total_attempts} 次")
    print(f"   成功次数: {total_success} 次")
    print(f"   失败次数: {total_attempts - total_success} 次")
    print(f"   成功率: {success_rate:.1f}%")
    print("")

    # Token 统计汇总
    total_all_tokens = sum(r["tokens"] for r in all_runs_results)
    total_all_prompt_tokens = sum(r["prompt_tokens"] for r in all_runs_results)
    total_all_completion_tokens = sum(r["completion_tokens"] for r in all_runs_results)
    avg_tokens_per_run = total_all_tokens / TOTAL_RUNS if TOTAL_RUNS > 0 else 0

    print("📊 Token 使用统计:")
    print(f"   输入 tokens: {total_all_prompt_tokens:,}")
    print(f"   输出 tokens: {total_all_completion_tokens:,}")
    print(f"   总计 tokens: {total_all_tokens:,}")
    print(f"   平均每次: {avg_tokens_per_run:,.0f} tokens")
    print("")

    # 最终评价
    if success_rate == 100:
        print("🎉 完美！所有运行100%成功！")
        return 0
    elif success_rate >= 80:
        print(f"✅ 良好！成功率达到 {success_rate:.1f}%")
        return 0
    elif success_rate >= 50:
        print(f"⚠️  一般，成功率 {success_rate:.1f}%，需要改进")
        return 1
    else:
        print(f"❌ 较差，成功率仅 {success_rate:.1f}%，需要重点优化")
        return 1


if __name__ == "__main__":
    sys.exit(run_gemmini_generator())
