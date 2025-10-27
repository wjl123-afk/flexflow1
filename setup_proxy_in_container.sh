#!/bin/bash

# ==========================================
# 🌐 在容器内设置代理环境变量
# ==========================================

echo "=== 在容器内配置代理 ==="

# 设置代理环境变量（172.17.0.1 是宿主机的 Docker 网关IP）
export http_proxy=http://172.17.0.1:7890
export https_proxy=http://172.17.0.1:7890
export HTTP_PROXY=http://172.17.0.1:7890
export HTTPS_PROXY=http://172.17.0.1:7890
export no_proxy=localhost,127.0.0.1,::1

# 设置 Git 代理（如果需要）
git config --global http.proxy http://172.17.0.1:7890
git config --global https.proxy http://172.17.0.1:7890

# 设置 pip 代理（如果需要）
pip config set global.proxy http://172.17.0.1:7890 2>/dev/null || true

# 显示配置
echo ""
echo "✅ 代理环境变量已设置"
echo "http_proxy: $http_proxy"
echo "https_proxy: $https_proxy"
echo ""
echo "🧪 测试代理连接..."
curl -I http://www.google.com 2>&1 | head -5
echo ""

