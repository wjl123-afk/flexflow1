#!/bin/bash

# ==========================================
# 🧹 清理僵尸进程和旧进程
# ==========================================

echo "=== 清理环境 ==="

# 1. 杀死所有 mihomo 进程
echo "🛑 停止所有 mihomo 进程..."
pkill -9 mihomo 2>/dev/null || true

# 2. 等待进程结束
sleep 2

# 3. 再次检查
echo "📊 检查剩余进程..."
if pgrep mihomo > /dev/null; then
    echo "⚠️  仍有 mihomo 进程运行："
    ps aux | grep mihomo | grep -v grep
else
    echo "✅ 所有 mihomo 进程已停止"
fi

echo ""
echo "现在可以重新启动 mihomo"

