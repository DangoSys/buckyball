import httpx
import json
import os
from dotenv import load_dotenv
import sys
import redis
from pathlib import Path
from typing import Optional, List, Dict

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from utils.stream_run import stream_run_logger
from utils.event_common import check_result
from steps.tools import create_code_agent_manager

load_dotenv()

config = {
    "type": "event",
    "name": "agent",
    "description": "通用 agent 处理器，根据 agentRole 加载不同系统 prompt，支持多种角色（code/design/test等）",
    "subscribes": ["agent.prompt"],
    "emits": ["agent.response"],
    "input": {
        "type": "object",
        "properties": {
            "agentRole": {"type": "string"},
            "promptPath": {"type": "string"},
            "systemPromptPath": {"type": "string"},
            "workDir": {"type": "string"},
            "model": {"type": "string"},
            "traceId": {"type": "string"},
            "apiKey": {"type": "string"},
            "baseUrl": {"type": "string"},
            "sessionId": {"type": "string"},
        },
    },
    "flows": ["agent"],
}

# ==================================================================================
# Redis 会话存储
# ==================================================================================


class SessionStore:
    """会话存储管理器，支持 Redis 和内存两种模式"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.memory_store: Dict[str, List[Dict]] = {}
        self.use_redis = False

        # 尝试连接 Redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_enabled = os.getenv("REDIS_ENABLED", "true").lower() == "true"

        if redis_enabled:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self.use_redis = True
                print(f"✅ Redis connected: {redis_url}")
            except Exception as e:
                print(f"⚠️  Redis connection failed: {e}, fallback to memory")
                self.use_redis = False
        else:
            print("ℹ️  Redis disabled, using memory storage")

    def get(self, session_id: str) -> Optional[List[Dict]]:
        """获取会话历史"""
        if self.use_redis and self.redis_client:
            try:
                data = self.redis_client.get(f"session:{session_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                print(f"Redis get error: {e}")

        return self.memory_store.get(session_id)

    def set(self, session_id: str, messages: List[Dict], ttl: int = 86400):
        """保存会话历史（默认 TTL 24 小时）"""
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    ttl,
                    json.dumps(messages, ensure_ascii=False),
                )
                return
            except Exception as e:
                print(f"Redis set error: {e}, fallback to memory")

        # Fallback to memory
        self.memory_store[session_id] = messages

    def exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        if self.use_redis and self.redis_client:
            try:
                return self.redis_client.exists(f"session:{session_id}") > 0
            except Exception as e:
                print(f"Redis exists error: {e}")

        return session_id in self.memory_store

    def delete(self, session_id: str):
        """删除会话"""
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.delete(f"session:{session_id}")
                return
            except Exception as e:
                print(f"Redis delete error: {e}")

        self.memory_store.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """列出所有会话 ID"""
        if self.use_redis and self.redis_client:
            try:
                keys = self.redis_client.keys("session:*")
                return [k.replace("session:", "") for k in keys]
            except Exception as e:
                print(f"Redis list error: {e}")

        return list(self.memory_store.keys())


# 全局会话存储实例
SESSION_STORE = SessionStore()

# 全局工具管理器
TOOL_MANAGER = create_code_agent_manager()


def get_default_system_prompt_path(agent_role: str) -> str:
    """根据 agent 角色获取默认系统 prompt 文件路径"""
    return os.path.join(
        os.path.dirname(__file__), "prompt", "agent", f"{agent_role}_agent.md"
    )


def replace_prompt_placeholders(content: str, work_dir: Optional[str] = None) -> str:
    """替换 prompt 中的占位符为实际路径"""
    # 从环境变量或参数获取工作目录
    if not work_dir:
        work_dir = os.getenv("WORK_DIR") or os.getenv("BUCKYBALL_WORK_DIR")

    work_dir_path = Path(work_dir)

    # 从环境变量获取构建脚本路径，或使用默认值
    build_script_path = str(work_dir_path / "scripts/build_gemmini.sh")
    build_log_path = str(work_dir_path / "build_logs/gemmini_build.log")

    # 替换占位符
    content = content.replace("{BUILD_SCRIPT_PATH}", build_script_path)
    content = content.replace("{BUILD_LOG_PATH}", build_log_path)
    content = content.replace("{WORK_DIR}", str(work_dir_path))

    return content


def load_system_prompt(
    agent_role: str,
    system_prompt_path: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> str:
    """从 markdown 文件加载系统 prompt，并替换占位符"""
    prompt_path = system_prompt_path or get_default_system_prompt_path(agent_role)

    if not os.path.isabs(prompt_path):
        prompt_path = os.path.abspath(prompt_path)

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"System prompt file is empty: {prompt_path}")

    # 替换占位符
    content = replace_prompt_placeholders(content, work_dir)

    return content


async def handler(input_data, context):
    agent_role = input_data.get("agentRole", "code")
    context.logger.info(f"{agent_role} agent - 开始处理", {"input": input_data})

    prompt_path = input_data.get("promptPath")
    system_prompt_path = input_data.get("systemPromptPath")
    work_dir = input_data.get("workDir", os.getcwd())
    model = input_data.get("model", "deepseek-chat")
    trace_id = input_data.get("traceId")
    session_id = input_data.get("sessionId")

    # API配置 - 严格检查
    api_key = (
        input_data.get("apiKey") or os.getenv("API_KEY") or os.getenv("LLM_API_KEY")
    )
    base_url = (
        input_data.get("baseUrl") or os.getenv("BASE_URL") or os.getenv("LLM_BASE_URL")
    )

    # 严格要求配置
    if not api_key:
        error_msg = (
            "❌ CRITICAL: API Key not configured!\n"
            "Please set one of the following:\n"
            "  - API_KEY environment variable\n"
            "  - LLM_API_KEY environment variable\n"
            "  - Pass 'apiKey' in request\n\n"
            "Example:\n"
            "  export API_KEY='your-api-key-here'\n"
        )
        context.logger.error(error_msg)
        await check_result(
            context, 1, continue_run=False, extra_fields={"error": error_msg}
        )
        return

    if not base_url:
        error_msg = (
            "❌ CRITICAL: Base URL not configured!\n"
            "Please set one of the following:\n"
            "  - BASE_URL environment variable\n"
            "  - LLM_BASE_URL environment variable\n"
            "  - Pass 'baseUrl' in request\n\n"
            "Example:\n"
            "  export BASE_URL='https://api.deepseek.com/v1'\n"
            "  export BASE_URL='https://api.anthropic.com/v1'\n"
        )
        context.logger.error(error_msg)
        await check_result(
            context, 1, continue_run=False, extra_fields={"error": error_msg}
        )
        return

    context.logger.info(f"使用 API: {base_url}, Key: {api_key[:10]}...")

    # 读取系统 prompt
    try:
        system_prompt = load_system_prompt(agent_role, system_prompt_path, work_dir)
        used_prompt_path = system_prompt_path or get_default_system_prompt_path(
            agent_role
        )
        context.logger.info(f"成功加载系统 prompt ({agent_role}): {used_prompt_path}")
    except Exception as e:
        error_msg = f"Failed to load system prompt for {agent_role}: {str(e)}"
        context.logger.error(error_msg)
        await check_result(
            context, 1, continue_run=False, extra_fields={"error": error_msg}
        )
        return

    # 读取用户 prompt 文件
    try:
        full_prompt_path = (
            os.path.join(work_dir, prompt_path)
            if not os.path.isabs(prompt_path)
            else prompt_path
        )

        if not os.path.exists(full_prompt_path):
            raise FileNotFoundError(f"User prompt file not found: {full_prompt_path}")

        with open(full_prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()

        context.logger.info(f"成功读取用户 prompt 文件: {full_prompt_path}")

    except Exception as e:
        error_msg = f"Failed to read user prompt file: {str(e)}"
        context.logger.error(error_msg)
        await check_result(
            context, 1, continue_run=False, extra_fields={"error": error_msg}
        )
        return

    # 获取或创建会话
    if session_id and SESSION_STORE.exists(session_id):
        messages = SESSION_STORE.get(session_id)
        context.logger.info(f"恢复会话 {session_id}，历史消息数: {len(messages)}")
        # 添加新的用户消息
        messages.append(
            {
                "role": "user",
                "content": f"Work Directory: {work_dir}\n\nNew Request:\n{prompt_content}",
            }
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Work Directory: {work_dir}\n\nRequest:\n{prompt_content}",
            },
        ]
        if session_id:
            context.logger.info(f"创建新会话 {session_id}")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    files_created = []
    files_read = []
    # 防止无限循环
    max_iterations = 10
    iteration = 0

    # 获取工具定义
    tools_schema = TOOL_MANAGER.get_tools_schema()

    try:
        async with httpx.AsyncClient() as client:

            while iteration < max_iterations:
                iteration += 1
                context.logger.info(f"==== 迭代 {iteration} ====")

                payload = {
                    "model": model,
                    "messages": messages,
                    "tools": tools_schema,
                    "temperature": 0.7,
                }

                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120.0,
                )

                if response.status_code != 200:
                    error_msg = f"API error: {response.status_code} - {response.text}"
                    context.logger.error(error_msg)
                    await check_result(
                        context,
                        1,
                        continue_run=False,
                        extra_fields={"error": error_msg},
                    )
                    return

                result = response.json()
                context.logger.info(
                    f"API 响应: {json.dumps(result, ensure_ascii=False)[:500]}"
                )

                # 检测 mock 响应
                response_id = result.get("id", "")
                if "mock" in response_id.lower():
                    error_msg = (
                        f"❌ CRITICAL: Detected mock/test response!\n"
                        f"Response ID: {response_id}\n\n"
                        f"This indicates the API is not properly configured or returning test data.\n"
                        f"Please check:\n"
                        f"  1. BASE_URL is correct: {base_url}\n"
                        f"  2. API_KEY is valid: {api_key[:20]}...\n"
                        f"  3. The API endpoint is responding with real LLM data\n\n"
                        f"If you see 'msg_mock_static', you are likely using a test/mock server.\n"
                    )
                    context.logger.error(error_msg)
                    await check_result(
                        context,
                        1,
                        continue_run=False,
                        extra_fields={"error": error_msg},
                    )
                    return

                if "choices" not in result:
                    error_msg = f"Invalid API response format: {json.dumps(result, ensure_ascii=False)}"
                    context.logger.error(error_msg)
                    await check_result(
                        context,
                        1,
                        continue_run=False,
                        extra_fields={"error": error_msg},
                    )
                    return

                assistant_message = result["choices"][0]["message"]

                # 添加助手响应到消息历史
                messages.append(assistant_message)

                # 检查是否有工具调用
                tool_calls = assistant_message.get("tool_calls")

                # 对于 master agent，检查是否只调用了 deepwiki 而没有调用 call_agent
                if agent_role == "master" and tool_calls:
                    # 统计工具调用类型
                    has_call_agent = False
                    has_deepwiki = False
                    for tc in tool_calls:
                        tool_name = tc.get("function", {}).get("name", "")
                        if tool_name == "call_agent":
                            has_call_agent = True
                        elif tool_name in ["deepwiki_ask", "deepwiki_read_wiki"]:
                            has_deepwiki = True

                    # 如果已经查询了多次 deepwiki 但还没调用 call_agent，需要警告
                    if has_deepwiki and not has_call_agent and iteration >= 4:
                        # 统计历史中 deepwiki 调用次数
                        deepwiki_count = 0
                        for msg in messages:
                            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                for tc in msg.get("tool_calls", []):
                                    if tc.get("function", {}).get("name") in [
                                        "deepwiki_ask",
                                        "deepwiki_read_wiki",
                                    ]:
                                        deepwiki_count += 1

                        # 如果已经查询了4次或更多 deepwiki，警告应该开始开发
                        if deepwiki_count >= 4:
                            context.logger.info(
                                f"⚠️ Master agent 已经查询了 {deepwiki_count} 次 Deepwiki，应该开始调用 call_agent"
                            )

                if not tool_calls:
                    # 检查是否需要强制执行
                    # 1. Master agent 需要强制执行
                    # 2. Spec/Code agent 如果没有调用 write_file，需要提醒

                    if agent_role == "master":
                        # 检查最近的工具返回是否包含错误
                        last_tool_result = None
                        for msg in reversed(messages):
                            if msg.get("role") == "tool":
                                last_tool_result = msg.get("content", "")
                                break

                        # 检查是否已经调用过 call_agent
                        has_call_agent = False
                        for msg in messages:
                            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                for tc in msg.get("tool_calls", []):
                                    if (
                                        tc.get("function", {}).get("name")
                                        == "call_agent"
                                    ):
                                        has_call_agent = True
                                        break
                            if has_call_agent:
                                break

                        # 检查是否是错误反馈
                        is_error_feedback = last_tool_result and (
                            "❌ 无法继续实现" in last_tool_result
                            or "需要先调用 spec_agent" in last_tool_result
                            or "spec.md 文件不存在" in last_tool_result
                            or "❌ 审查不通过" in last_tool_result
                        )

                        # 前5轮且还没调用过 call_agent，或收到错误反馈后，强制执行
                        if (iteration <= 5 and not has_call_agent) or is_error_feedback:
                            context.logger.info(
                                f"⚠️ Master agent 第{iteration}轮没有工具调用"
                                + (
                                    " (收到错误反馈)"
                                    if is_error_feedback
                                    else " (还未调用 call_agent)"
                                )
                                + "，注入强制提示"
                            )

                            # 根据错误类型生成不同的提示
                            if (
                                is_error_feedback
                                and "需要先调用 spec_agent" in last_tool_result
                            ):
                                force_message = """⚠️ Code agent 返回错误：spec.md 不存在！

你必须先调用 spec_agent 编写 spec.md，然后再调用 code_agent。

立即行动：
call_agent(
  agent_role="spec",
  task_description="为当前 Ball 编写 spec.md，参考 arch/src/main/scala/prototype/nagisa/gelu/spec.md",
  context_files=["arch/src/main/scala/prototype/nagisa/gelu/spec.md"]
)

不要只返回文本说明，必须调用工具！"""
                            elif (
                                is_error_feedback
                                and "❌ 审查不通过" in last_tool_result
                            ):
                                force_message = """⚠️ Review agent 返回审查不通过！

根据 review_agent 的反馈，调用 code_agent 修复问题。

不要只返回文本说明，必须调用工具！"""
                            else:
                                # 根据迭代次数增强强制程度
                                if iteration >= 4:
                                    force_message = f"""🚨 CRITICAL ERROR 🚨

You have completed {iteration} iterations but NEVER called call_agent tool!

YOUR ONLY JOB IS TO CALL call_agent!

Execute this EXACT function call NOW (copy this JSON exactly):

{{"role": "assistant", "tool_calls": [{{"id": "force_call_1", "type": "function", "function": {{"name": "call_agent", "arguments": "{{\\"agent_role\\": \\"spec\\", \\"task_description\\": \\"为第一个 Ball 编写 spec.md\\", \\"context_files\\": [\\"arch/src/main/scala/prototype/nagisa/gelu/spec.md\\"]}}"}}}}]}}

DO NOT return text "-" or any explanation.
ONLY call the call_agent function.
This is iteration {iteration}. If you return text again, the system will fail."""
                                else:
                                    force_message = f"""⚠️ 你已经完成了{iteration}轮信息收集，现在必须开始实际开发！

你必须调用 call_agent 工具，不是返回文本说明！

示例调用格式：
```json
{{
  "tool_calls": [{{
    "function": {{
      "name": "call_agent",
      "arguments": {{
        "agent_role": "spec",
        "task_description": "为第一个 Ball 编写 spec.md",
        "context_files": ["arch/src/main/scala/prototype/nagisa/gelu/spec.md"]
      }}
    }}
  }}]
}}
```

不要再返回 "-" 或任何文本，必须调用工具！"""

                            messages.append(
                                {
                                    "role": "user",
                                    "content": force_message,
                                }
                            )
                            # 继续下一轮迭代
                            continue
                        else:
                            # 已经超过5轮且调用过 call_agent，或没有错误反馈，正常结束
                            final_response = assistant_message.get("content", "")
                            if has_call_agent:
                                context.logger.info(f"任务完成: {final_response}")
                            else:
                                context.logger.warning(
                                    "Master agent 超过5轮仍未调用 call_agent，强制结束"
                                )
                            break
                    else:
                        # 非 master agent，检查是否应该使用工具
                        final_response = assistant_message.get("content", "")

                        # 检查 spec/code agent 是否使用了 write_file
                        if agent_role in ["spec", "code"] and iteration <= 5:
                            # 检查历史消息中是否有 write_file 调用
                            has_write_file = False
                            for msg in messages:
                                if msg.get("role") == "assistant" and msg.get(
                                    "tool_calls"
                                ):
                                    for tc in msg.get("tool_calls", []):
                                        if (
                                            tc.get("function", {}).get("name")
                                            == "write_file"
                                        ):
                                            has_write_file = True
                                            break
                                if has_write_file:
                                    break

                            if not has_write_file:
                                context.logger.info(
                                    f"⚠️ {agent_role} agent 还没有调用 write_file，注入提醒"
                                )
                                messages.append(
                                    {
                                        "role": "user",
                                        "content": """⚠️ 你还没有使用 write_file 工具创建文件！

你必须使用 write_file 工具将内容写入文件，而不是只返回文本。

立即执行：
write_file(
  path="<目标文件路径>",
  content="<你编写的内容>"
)

不要只返回文本，必须调用 write_file 工具！""",
                                    }
                                )
                                # 继续下一轮迭代
                                continue

                        # 正常结束
                        context.logger.info(f"任务完成: {final_response}")
                        break

                # 执行所有工具调用
                context.logger.info(f"执行 {len(tool_calls)} 个工具调用")

                for tool_call in tool_calls:
                    tool_id = tool_call["id"]
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]

                    context.logger.info(f"调用工具: {tool_name}")
                    args_preview = (
                        tool_args[:300]
                        if isinstance(tool_args, str)
                        else json.dumps(tool_args, ensure_ascii=False)[:300]
                    )
                    context.logger.info(f"工具参数: {args_preview}")

                    # 使用工具管理器执行工具
                    tool_result = TOOL_MANAGER.execute_tool(
                        tool_name=tool_name,
                        arguments=tool_args,
                        work_dir=work_dir,
                        logger=context.logger,
                        # 传递当前使用的 model
                        model=model,
                    )

                    # 记录工具返回结果
                    result_preview = str(tool_result)[:500] if tool_result else "None"
                    context.logger.info(f"工具返回 ({tool_name}): {result_preview}")

                    # 记录操作
                    if tool_name == "write_file":
                        args = (
                            json.loads(tool_args)
                            if isinstance(tool_args, str)
                            else tool_args
                        )
                        files_created.append(args.get("path"))
                    elif tool_name == "read_file":
                        args = (
                            json.loads(tool_args)
                            if isinstance(tool_args, str)
                            else tool_args
                        )
                        files_read.append(args.get("path"))

                    # 添加工具结果到消息历史
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": tool_result,
                        }
                    )

                # 继续下一轮，让 AI 根据工具结果继续决策

            # 保存会话（TTL 24小时）
            if session_id:
                # 默认 24 小时
                session_ttl = int(os.getenv("SESSION_TTL", "86400"))
                SESSION_STORE.set(session_id, messages, ttl=session_ttl)
                context.logger.info(
                    f"会话 {session_id} 已保存，消息数: {len(messages)}, TTL: {session_ttl}s"
                )

            # 获取最终响应
            final_content = ""
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_content = msg.get("content")
                    break

            # 发送完整响应
            await context.emit(
                {
                    "topic": "agent.response",
                    "data": {
                        "agentRole": agent_role,
                        "response": final_content,
                        "files_created": files_created,
                        "files_read": files_read,
                        "iterations": iteration,
                        "traceId": trace_id,
                    },
                }
            )

            context.logger.info(
                f"{agent_role} agent 处理完成",
                {
                    "iterations": iteration,
                    "files_created": len(files_created),
                    "files_read": len(files_read),
                    "traceId": trace_id,
                },
            )

        # 返回结果
        await check_result(
            context,
            0,
            continue_run=False,
            extra_fields={
                "agentRole": agent_role,
                "response": final_content,
                "files": files_created,
                "filesRead": files_read,
                "iterations": iteration,
            },
        )

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        context.logger.error(f"{agent_role} agent 执行失败: {str(e)}")
        context.logger.error(f"堆栈跟踪:\n{error_traceback}")

        await check_result(
            context,
            1,
            continue_run=False,
            extra_fields={"error": str(e), "traceback": error_traceback},
        )

    return
