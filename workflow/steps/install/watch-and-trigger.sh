w#!/bin/bash
# 实时监听并触发Chipyard初始化

echo "🔍 设置实时日志监听..."
echo "⚡ 准备触发Chipyard初始化..."

# 清理旧的监听
pkill -f "tail.*Init Chipyard" 2>/dev/null || true

# 在后台启动日志监听
(
    echo "📡 开始监听服务端日志..."
    while IFS= read -r line; do
        if [[ "$line" == *"Init Chipyard"* ]] || [[ "$line" == *"🚀"* ]] || [[ "$line" == *"✅"* ]] || [[ "$line" == *"❌"* ]] || [[ "$line" == *"Processing"* ]]; then
            echo "🔄 $line"
        fi
    done < <(tail -f /tmp/motia.log 2>/dev/null)
) &

LISTENER_PID=$!

# 给监听器一点时间启动
sleep 1

echo "🚀 触发Chipyard初始化..."
curl -s -X POST http://localhost:5000/install \
  -H "Content-Type: application/json" \
  -d '{}' \
  -w "\n✅ 请求已发送，查看上方实时输出\n"

# 等待30秒后清理监听器
sleep 30
kill $LISTENER_PID 2>/dev/null || true