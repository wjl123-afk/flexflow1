#!/bin/bash
# CUDA 12.4 升级到 12.8 脚本
# 适用于 Ubuntu 22.04 系统
# 用于配置 vllm

set -e  # 遇到错误立即退出

echo "========================================="
echo "🚀 CUDA 12.4 升级到 12.8"
echo "========================================="
echo ""

# 检查系统版本
if [ ! -f /etc/os-release ]; then
    echo "❌ 无法检测系统版本"
    exit 1
fi

. /etc/os-release
echo "📋 系统信息: $NAME $VERSION"
echo ""

# 检查是否为Ubuntu 22.04
if [[ "$VERSION_ID" != "22.04" ]]; then
    echo "⚠️  警告: 此脚本针对 Ubuntu 22.04 优化"
    echo "   当前系统版本: $VERSION_ID"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查root权限
if [ "$EUID" -ne 0 ]; then 
    echo "❌ 请使用 sudo 运行此脚本"
    exit 1
fi

echo "========================================="
echo "📥 步骤 1: 下载 CUDA 12.8 仓库配置文件"
echo "========================================="

# 下载 pin 文件
echo "下载 cuda-ubuntu2204.pin..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin -O /tmp/cuda-ubuntu2204.pin

# 移动到正确位置
echo "配置 CUDA 仓库优先级..."
mv /tmp/cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

echo "✅ Pin 文件配置完成"
echo ""

echo "========================================="
echo "📦 步骤 2: 下载 CUDA 12.8 安装包"
echo "========================================="

# CUDA 12.8 版本信息
CUDA_VERSION="12.8.0"
CUDA_BUILD="570.86.10"
CUDA_DEB="cuda-repo-ubuntu2204-12-8-local_${CUDA_VERSION}-${CUDA_BUILD}-1_amd64.deb"
CUDA_DEB_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/${CUDA_DEB}"

echo "下载 CUDA 12.8 安装包..."
echo "文件名: $CUDA_DEB"
echo "URL: $CUDA_DEB_URL"
echo ""

cd /tmp
if [ -f "$CUDA_DEB" ]; then
    echo "⚠️  安装包已存在，跳过下载"
else
    wget "$CUDA_DEB_URL" -O "$CUDA_DEB"
    echo "✅ 下载完成"
fi
echo ""

echo "========================================="
echo "🔧 步骤 3: 安装 CUDA 12.8"
echo "========================================="

# 安装 deb 包
echo "安装 CUDA 仓库配置..."
dpkg -i "/tmp/$CUDA_DEB" || true

# 复制 keyring
echo "配置 GPG keyring..."
cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ || true

# 更新 apt
echo "更新软件包列表..."
apt-get update

# 安装 CUDA Toolkit 12.8
echo "安装 CUDA Toolkit 12-8..."
apt-get -y install cuda-toolkit-12-8

echo "✅ CUDA 12.8 安装完成"
echo ""

echo "========================================="
echo "⚙️  步骤 4: 配置环境变量"
echo "========================================="

# 创建环境变量配置脚本
cat > /etc/profile.d/cuda-12.8.sh << 'EOF'
# CUDA 12.8 环境变量配置
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

echo "✅ 环境变量配置文件已创建: /etc/profile.d/cuda-12.8.sh"
echo ""

# 应用环境变量（当前会话）
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "========================================="
echo "✅ 步骤 5: 验证安装"
echo "========================================="

# 检查 CUDA 版本
if [ -f "$CUDA_HOME/bin/nvcc" ]; then
    echo "NVCC 版本:"
    $CUDA_HOME/bin/nvcc --version
    echo ""
else
    echo "⚠️  警告: 无法找到 nvcc"
fi

# 检查 nvidia-smi（如果驱动已安装）
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA 驱动信息:"
    nvidia-smi
    echo ""
fi

echo "========================================="
echo "🎉 CUDA 12.8 升级完成！"
echo "========================================="
echo ""
echo "📝 重要提示:"
echo "1. 请重新登录或运行以下命令使环境变量生效:"
echo "   source /etc/profile.d/cuda-12.8.sh"
echo ""
echo "2. 或者手动设置:"
echo "   export CUDA_HOME=/usr/local/cuda-12.8"
echo "   export PATH=\$CUDA_HOME/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "3. 验证 CUDA 版本:"
echo "   nvcc --version"
echo ""

