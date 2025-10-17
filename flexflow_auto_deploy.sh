#!/bin/bash

################################################################################
# FlexFlow 完整环境自动化部署脚本
# 功能：在新的 CUDA 12.4 机器上一键部署 FlexFlow 推理环境
# 作者：BlockElite 研发团队
# 日期：2025.10.07
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 分隔线函数
print_separator() {
    echo -e "${BLUE}===================================================================${NC}"
}

################################################################################
# 第0步：环境检查与系统配置
################################################################################
check_environment() {
    print_separator
    log_info "第0步：环境检查与系统配置"
    print_separator
    
    # 检查操作系统
    log_info "检查操作系统版本..."
    uname -a
    
    # 检查GPU
    log_info "检查GPU可用性..."
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "未检测到 NVIDIA GPU 驱动，请先安装 CUDA 12.4 驱动"
        exit 1
    fi
    nvidia-smi
    
    # 检查内存
    log_info "检查系统内存..."
    free -h
    
    # 检查存储空间
    log_info "检查存储空间..."
    df -h
    
    log_success "环境检查完成"
}

################################################################################
# 第1步：配置 mihomo 代理（加速安装）
################################################################################
setup_mihomo_proxy() {
    print_separator
    log_info "第1步：配置 mihomo 代理"
    print_separator
    
    # 安装基础工具
    log_info "安装基础工具 (sshpass, curl)..."
    apt-get update
    apt-get install -y sshpass curl
    
    # 创建目录并下载 mihomo
    log_info "从SFTP服务器获取 mihomo 代理工具..."
    mkdir -p /opt/mihomo
    cd /opt/mihomo
    
    sshpass -p 'Yn783CWe' sftp -P 15022 -o StrictHostKeyChecking=no 15256911585@pan.blockelite.cn << 'EOF'
cd mihomo
get mihomo
get mihomo_config.yaml
quit
EOF
    
    # 启动 mihomo 代理
    log_info "启动 mihomo 代理服务..."
    chmod +x mihomo
    nohup ./mihomo -f mihomo_config.yaml > mihomo.log 2>&1 &
    sleep 3
    
    # 验证代理连接
    log_info "验证代理连接..."
    if curl -I --proxy http://127.0.0.1:7890 http://www.google.com 2>&1 | grep -q "200\|301\|302"; then
        log_success "mihomo 代理配置完成，网络访问已加速！"
    else
        log_warning "代理连接验证失败，但继续执行..."
    fi
}

################################################################################
# 第2步：安装 Docker Engine
################################################################################
install_docker() {
    print_separator
    log_info "第2步：安装 Docker Engine"
    print_separator
    
    # 检查是否已安装 Docker
    if command -v docker &> /dev/null; then
        log_warning "Docker 已安装，跳过安装步骤"
        docker --version
        return
    fi
    
    # 快速安装 Docker Engine (尝试使用代理，失败则直连)
    log_info "使用官方脚本安装 Docker..."
    if curl -x http://127.0.0.1:7890 --connect-timeout 5 -fsSL https://get.docker.com | sh; then
        log_success "通过代理安装 Docker 成功"
    else
        log_warning "代理安装失败，尝试直连下载..."
        curl -fsSL https://get.docker.com | sh
    fi
    
    # 启动 Docker 服务
    log_info "启动 Docker 服务..."
    systemctl start docker
    systemctl enable docker
    
    # 验证安装
    log_info "验证 Docker 安装..."
    docker --version
    docker run hello-world
    
    log_success "Docker Engine 安装完成"
}

################################################################################
# 第3步：安装 NVIDIA Container Toolkit
################################################################################
install_nvidia_toolkit() {
    print_separator
    log_info "第3步：安装 NVIDIA Container Toolkit"
    print_separator
    
    # 添加软件源 GPG 密钥 (尝试使用代理，失败则直连)
    log_info "添加 NVIDIA Container Toolkit 软件源..."
    if ! curl -x http://127.0.0.1:7890 --connect-timeout 5 -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg; then
        log_warning "代理下载失败，尝试直连..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    fi
    
    # 添加软件源
    if ! curl -x http://127.0.0.1:7890 --connect-timeout 5 -sSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list; then
        log_warning "代理下载失败，尝试直连..."
        curl -sSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    fi
    
    # 更新软件包索引
    apt-get update
    
    # 安装 NVIDIA Container Toolkit
    log_info "安装 NVIDIA Container Toolkit..."
    apt-get install -y nvidia-container-toolkit
    
    # 验证安装版本
    log_info "验证安装版本..."
    nvidia-ctk --version
    
    # 配置 Docker 使用 NVIDIA 运行时
    log_info "配置 Docker 使用 NVIDIA 运行时..."
    nvidia-ctk runtime configure --runtime=docker
    
    # 重启 Docker 服务
    log_info "重启 Docker 服务使配置生效..."
    systemctl restart docker
    
    log_success "NVIDIA Container Toolkit 安装完成"
}

################################################################################
# 第4步：配置私有 Registry 访问
################################################################################
configure_registry() {
    print_separator
    log_info "第4步：配置私有 Registry 访问"
    print_separator
    
    # 创建 Docker 配置目录
    mkdir -p /etc/docker
    
    # 生成完整配置
    log_info "生成 Docker 配置文件..."
    cat > /etc/docker/daemon.json << 'EOF'
{
  "insecure-registries": [
    "js3.blockelite.cn:12570",
    "localhost:12570"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "runtimes": {
    "nvidia": {
      "args": [],
      "path": "nvidia-container-runtime"
    }
  }
}
EOF
    
    # 重启 Docker 应用配置
    log_info "重启 Docker 应用配置..."
    systemctl restart docker
    
    # 验证配置文件
    log_info "验证配置文件..."
    apt-get install -y jq
    cat /etc/docker/daemon.json | jq .
    
    # Registry 认证
    log_info "登录私有 Registry..."
    echo 'BlockElite2024' | docker login js3.blockelite.cn:12570 -u admin --password-stdin
    
    # 测试 Registry 连通性
    log_info "测试 Registry 连通性..."
    curl -u admin:BlockElite2024 -s http://js3.blockelite.cn:12570/v2/_catalog | jq .
    
    log_success "私有 Registry 配置完成"
}

################################################################################
# 第5步：拉取 FlexFlow CUDA 镜像
################################################################################
pull_flexflow_image() {
    print_separator
    log_info "第5步：拉取 FlexFlow CUDA 镜像"
    print_separator
    
    # 查看可用标签
    log_info "查看可用镜像标签..."
    curl -u admin:BlockElite2024 -s http://js3.blockelite.cn:12570/v2/cuda124-mihomo-python-ubuntu2204/tags/list | jq .
    
    # 拉取 CUDA 12.4 FlexFlow 镜像
    log_info "拉取 CUDA 12.4 FlexFlow 镜像（约9.33GB，请耐心等待）..."
    docker pull js3.blockelite.cn:12570/cuda124-mihomo-python-ubuntu2204:flexflow
    
    # 验证镜像拉取
    log_info "验证镜像拉取结果..."
    docker images | grep cuda124
    
    log_success "FlexFlow 镜像拉取完成"
}

################################################################################
# 第6步：下载 FlexFlow 源码包
################################################################################
download_flexflow_source() {
    print_separator
    log_info "第6步：下载 FlexFlow 源码包"
    print_separator
    
    # 创建工作空间目录
    log_info "创建工作空间目录..."
    mkdir -p /workspace
    cd /workspace
    
    # 从 SFTP 下载源码包
    log_info "从 SFTP 服务器下载 FlexFlow 源码包..."
    sshpass -p 'Yn783CWe' sftp -P 15022 -o StrictHostKeyChecking=no 15256911585@pan.blockelite.cn << 'EOF'
cd mihomo
get flexflow-serve-20250919_153818.tar.gz
quit
EOF
    
    # 解压源码包
    log_info "解压 FlexFlow 源码包..."
    tar -xzvf flexflow-serve-20250919_153818.tar.gz
    
    # 验证解压结果
    if [ -d "/workspace/flexflow-serve" ]; then
        log_success "FlexFlow 源码包下载并解压完成"
        ls -la /workspace/flexflow-serve
    else
        log_error "FlexFlow 源码包解压失败"
        exit 1
    fi
}

################################################################################
# 第7步：启动容器并绑定工作空间
################################################################################
start_container() {
    print_separator
    log_info "第7步：启动容器并绑定工作空间"
    print_separator
    
    # 检查是否已存在同名容器
    if docker ps -a | grep -q cuda-mihomo; then
        log_warning "检测到已存在的 cuda-mihomo 容器，正在删除..."
        docker stop cuda-mihomo 2>/dev/null || true
        docker rm cuda-mihomo 2>/dev/null || true
    fi
    
    # 启动容器
    log_info "启动 FlexFlow GPU 容器..."
    docker run -d --name cuda-mihomo \
        --gpus all \
        --shm-size=2g \
        -v /workspace:/workspace \
        -p 7892:7890 -p 8888:8888 \
        js3.blockelite.cn:12570/cuda124-mihomo-python-ubuntu2204:flexflow \
        tail -f /dev/null
    
    # 等待容器启动
    sleep 3
    
    # 验证容器运行状态
    if docker ps | grep -q cuda-mihomo; then
        log_success "容器启动成功"
        docker ps | grep cuda-mihomo
    else
        log_error "容器启动失败"
        exit 1
    fi
}

################################################################################
# 第8步：容器内配置环境并运行测试
################################################################################
run_flexflow_test() {
    print_separator
    log_info "第8步：容器内配置环境并运行 FlexFlow 推理测试"
    print_separator
    
    # 创建容器内执行脚本
    log_info "创建容器内测试脚本..."
    cat > /tmp/flexflow_test.sh << 'EOF'
#!/bin/bash
set -e

echo "===================================================================="
echo "[INFO] 激活 FlexFlow conda 环境..."
echo "===================================================================="
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate flexflow
python --version

echo "===================================================================="
echo "[INFO] 进入 FlexFlow 构建目录..."
echo "===================================================================="
cd /workspace/flexflow-serve/build

echo "===================================================================="
echo "[INFO] 读取并执行 Python 环境配置..."
echo "===================================================================="
if [ -f "set_python_envs.sh" ]; then
    cat set_python_envs.sh
    echo ""
    echo "[INFO] 执行环境变量配置..."
    source set_python_envs.sh
else
    echo "[ERROR] set_python_envs.sh 文件不存在！"
    exit 1
fi

echo "===================================================================="
echo "[INFO] 验证环境变量配置..."
echo "===================================================================="
echo "PYTHONPATH: $PYTHONPATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo "===================================================================="
echo "[INFO] 切换到示例目录..."
echo "===================================================================="
cd /workspace/flexflow-serve/examples

echo "===================================================================="
echo "[INFO] 运行 FlexFlow 推理测试..."
echo "===================================================================="
python ff.py

echo "===================================================================="
echo "[SUCCESS] FlexFlow 推理测试完成！"
echo "===================================================================="
EOF
    
    # 复制脚本到容器内
    docker cp /tmp/flexflow_test.sh cuda-mihomo:/tmp/flexflow_test.sh
    
    # 在容器内执行测试（不使用 -it 避免 TTY 问题）
    log_info "在容器内执行 FlexFlow 推理测试..."
    docker exec cuda-mihomo bash /tmp/flexflow_test.sh
    
    log_success "FlexFlow 推理测试执行完成"
}

################################################################################
# 主函数
################################################################################
main() {
    echo ""
    print_separator
    log_info "FlexFlow 完整环境自动化部署开始"
    print_separator
    echo ""
    
    # 执行所有步骤
    check_environment
    setup_mihomo_proxy
    install_docker
    install_nvidia_toolkit
    configure_registry
    pull_flexflow_image
    download_flexflow_source
    start_container
    run_flexflow_test
    
    echo ""
    print_separator
    log_success "🎉 FlexFlow 环境部署和测试全部完成！"
    print_separator
    echo ""
    
    # 打印后续使用说明
    cat << 'EOF'
📋 后续使用说明：

1. 进入容器进行开发：
   docker exec -it cuda-mihomo bash

2. 查看容器日志：
   docker logs cuda-mihomo

3. 停止容器：
   docker stop cuda-mihomo

4. 启动容器：
   docker start cuda-mihomo

5. FlexFlow 项目路径：
   宿主机：/workspace/flexflow-serve
   容器内：/workspace/flexflow-serve

6. Web UI 访问（如果启动了服务）：
   http://YOUR_SERVER_IP:8888

7. Git LFS 服务器（管理 FlexFlow 模型文件）：
   http://js3.blockelite.cn:12572/admin/flexflow-serve
   用户名：admin
   密码：admin123

祝开发顺利！🚀
EOF
}

# 执行主函数
main "$@"

