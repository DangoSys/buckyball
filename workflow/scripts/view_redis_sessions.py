#!/usr/bin/env python3
"""查看 Redis 中的会话数据

Usage:
    # 简要信息
    python view_redis_sessions.py
    # 详细信息
    python view_redis_sessions.py -v
    # 详细信息
    python view_redis_sessions.py --verbose
"""

import redis
import json
import os
import sys
from datetime import datetime


def format_timestamp(seconds):
    """格式化时间戳"""
    if seconds < 0:
        return "已过期"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs}s"


def main():
    # 检查命令行参数
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    try:
        r = redis.from_url(redis_url, decode_responses=True)
        r.ping()
        print("✅ Redis 连接成功:", redis_url)
        print()
    except Exception as e:
        print(f"❌ Redis 连接失败: {e}")
        sys.exit(1)

    # 获取所有 session 键
    keys = r.keys("session:*")

    if not keys:
        print("📭 没有找到会话数据")
        return

    print(f"📊 找到 {len(keys)} 个会话\n")
    print("=" * 100)

    for i, key in enumerate(keys, 1):
        session_id = key.replace("session:", "")
        ttl = r.ttl(key)
        data = r.get(key)

        print(f"\n[{i}] Session ID: {session_id}")
        print(f"    TTL: {format_timestamp(ttl)}")

        if not data:
            print("    ⚠️  空会话")
            continue

        try:
            messages = json.loads(data)
            print(f"    消息数: {len(messages)}")

            # 统计角色
            role_count = {}
            for msg in messages:
                role = msg.get("role", "unknown")
                role_count[role] = role_count.get(role, 0) + 1

            print(f"    角色统计: {dict(role_count)}")

            # 显示详细内容（如果指定了 verbose 参数）
            if verbose:
                print("\n    " + "─" * 96)
                for j, msg in enumerate(messages, 1):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])

                    print(f"\n    [{j}] Role: {role}")

                    if content:
                        # 限制内容长度
                        if len(content) > 500:
                            print(
                                f"    Content: {content[:500]}...\n             (共 {len(content)} 字符)"
                            )
                        else:
                            print(f"    Content: {content}")

                    if tool_calls:
                        print(f"    Tool Calls: {len(tool_calls)} 个")
                        for tc in tool_calls:
                            func_name = tc.get("function", {}).get("name", "unknown")
                            print(f"      - {func_name}")

                print("\n    " + "─" * 96)

        except json.JSONDecodeError as e:
            print(f"    ⚠️  JSON 解析失败: {e}")
            print(f"    原始数据: {data[:200]}...")

        print("\n" + "=" * 100)

    print(f"\n✨ 共 {len(keys)} 个会话")


if __name__ == "__main__":
    main()
