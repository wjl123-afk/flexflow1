#!/bin/bash

# ==========================================
# 🔧 在宿主机上设置 mihomo 代理
# ==========================================

set -e

echo "=== 在宿主机上配置 mihomo 代理 ==="

# 1. 安装依赖
echo "📦 安装依赖..."
apt-get update && apt-get install -y sshpass

# 2. 创建 mihomo 目录
echo "📁 创建目录..."
mkdir -p /opt/mihomo && cd /opt/mihomo

# 3. 检查是否已存在文件
if [ -f "mihomo" ] && [ -f "mihomo_config.yaml" ]; then
    echo "✅ mihomo 文件已存在，跳过下载"
else
    echo "⬇️  通过 SFTP 下载 mihomo..."
    sshpass -p 'Yn783CWe' sftp -P 15022 -o StrictHostKeyChecking=no 15256911585@pan.blockelite.cn << 'EOF'
cd mihomo
get mihomo
get mihomo_config.yaml
quit
EOF
fi

# 4. 赋予执行权限
echo "🔧 设置权限..."
chmod +x mihomo

# 5. 停止旧的 mihomo 进程（如果存在）
echo "🛑 停止旧进程..."
pkill -9 mihomo 2>/dev/null || true

# 6. 启动 mihomo
echo "🚀 启动 mihomo..."
nohup ./mihomo -f mihomo_config.yaml > mihomo.log 2>&1 &

# 7. 等待启动
sleep 5

# 8. 检查状态
echo ""
echo "📊 检查运行状态..."
ps aux | grep mihomo | grep -v grep

echo ""
echo "📋 最近日志："
tail -30 mihomo.log

# 9. 测试连接
echo ""
echo "🧪 测试代理连接..."
curl -I --proxy http://127.0.0.1:7890 http://www.google.com 2>&1 | head -3 || echo "⚠️  代理可能无法连接到上游节点"

echo ""
echo "✅ mihomo 代理已在宿主机上启动！"
echo "📍 容器可通过 http://172.17.0.1:7890 访问代理"

