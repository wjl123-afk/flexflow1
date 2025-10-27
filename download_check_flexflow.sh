#!/bin/bash

# 下载check_flexflow_env.sh到容器的脚本
# 在宿主机上运行：bash download_check_flexflow.sh

echo "📥 开始下载check_flexflow_env.sh到容器..."

# 检查容器是否在运行
if ! docker ps | grep -q cuda-mihomo; then
    echo "❌ 容器cuda-mihomo未运行"
    exit 1
fi

# 复制脚本到容器
docker cp check_flexflow_env.sh cuda-mihomo:/workspace/

# 给脚本添加执行权限
docker exec cuda-mihomo chmod +x /workspace/check_flexflow_env.sh

echo "✅ 脚本已下载并设置权限"
echo ""
echo "在容器内使用方法:"
echo "  1. 进入容器: docker exec -it cuda-mihomo bash"
echo "  2. 执行脚本: source /workspace/check_flexflow_env.sh"
echo "  或直接执行: bash /workspace/check_flexflow_env.sh"

