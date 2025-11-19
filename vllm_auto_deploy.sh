#!/bin/bash

################################################################################
# vLLM 完整环境自动化部署脚本
# 功能：在新的 CUDA 12.4 机器上一键部署 vLLM 推理环境
# 作者：BlockElite 研发团队（改编版）
# 日期：2025.11.19
################################################################################

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 日志函数
log_info()    { echo -e "${BLUE}[INFO]${NC}    $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC}   $1"; }
print_sep()   { echo -e "${BLUE}===================================================================${NC}"; }

# 全局变量
VLLM_IMAGE="${VLLM_IMAGE:-nvidia/cuda:12.4.0-devel-ubuntu22.04}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-gpu}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
SCRIPT_DIR="$WORKSPACE_DIR/scripts"
HOST_HTTP_PROXY="${http_proxy:-}"
HOST_HTTPS_PROXY="${https_proxy:-}"

################################################################################
# 第0步：环境检查
################################################################################
check_environment() {
    print_sep
    log_info "第0步：环境检查与系统配置"
    print_sep

    log_info "检查操作系统内核"
    uname -a

    log_info "检查 GPU 与驱动版本（需 >= 550，推荐 565+）"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_error "未检测到 nvidia-smi，请先安装 NVIDIA 驱动 (CUDA 12.4+)"
        exit 1
    fi
    nvidia-smi

    log_info "检查可用磁盘空间"
    df -h "$WORKSPACE_DIR" || df -h

    log_info "检查网络连通性"
    ping -c 2 mirrors.tuna.tsinghua.edu.cn || log_warning "无法 ping 通清华镜像，可忽略"

    log_success "环境检查完成"
}

################################################################################
# 第1步：配置 mihomo 代理（可选）
################################################################################
setup_mihomo_proxy() {
    print_sep
    log_info "第1步：配置 mihomo 代理（如已运行可跳过）"
    print_sep

    if pgrep -f "mihomo" >/dev/null 2>&1; then
        log_warning "检测到已有 mihomo 进程，跳过重新下载"
        return
    fi

    apt-get update
    apt-get install -y sshpass curl

    mkdir -p /opt/mihomo && cd /opt/mihomo
    log_info "通过 SFTP 拉取 mihomo 与配置..."
    sshpass -p 'Yn783CWe' sftp -P 15022 -o StrictHostKeyChecking=no 15256911585@pan.blockelite.cn <<'EOF'
cd mihomo
get mihomo
get mihomo_config.yaml
quit
EOF

    chmod +x ./mihomo
    nohup ./mihomo -f mihomo_config.yaml >mihomo.log 2>&1 &
    sleep 3

    if curl -I --proxy http://127.0.0.1:7890 http://www.google.com >/dev/null 2>&1; then
        log_success "mihomo 代理可用"
    else
        log_warning "代理连通性验证失败，请自行确认 127.0.0.1:7890"
    fi
}

################################################################################
# 第2步：安装 Docker
################################################################################
install_docker() {
    print_sep
    log_info "第2步：安装 Docker Engine"
    print_sep

    if command -v docker >/dev/null 2>&1; then
        log_warning "Docker 已存在，跳过安装"
        docker --version
        return
    fi

    curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu $(lsb_release -cs) stable" \
        >/etc/apt/sources.list.d/docker.list

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io

    systemctl enable --now docker
    docker --version
    log_success "Docker 安装完成"
}

################################################################################
# 第3步：安装 NVIDIA Container Toolkit
################################################################################
install_nvidia_toolkit() {
    print_sep
    log_info "第3步：安装 NVIDIA Container Toolkit"
    print_sep

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
        >/etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit

    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker

    log_success "NVIDIA Container Toolkit 安装完成"
}

################################################################################
# 第4步：准备工作目录
################################################################################
prepare_workspace() {
    print_sep
    log_info "第4步：准备 /workspace 目录与脚本目录"
    print_sep

    mkdir -p "$WORKSPACE_DIR"/{models,configs,logs,scripts}
    chmod -R 777 "$WORKSPACE_DIR"

    cat >"$SCRIPT_DIR/README.txt" <<EOF
该目录由 vLLM 自动部署脚本生成：
- run_vllm_server.sh  启动 vLLM OpenAI 兼容服务
- bench_vllm_client.sh 使用 vLLM benchmark 工具压测
默认端口：8015，可通过环境变量 VLLM_PORT 覆盖
EOF

    log_success "工作目录准备完成：$WORKSPACE_DIR"
}

################################################################################
# 第5步：拉取 vLLM 基础镜像
################################################################################
pull_vllm_image() {
    print_sep
    log_info "第5步：拉取 CUDA 基础镜像 $VLLM_IMAGE"
    print_sep

    docker pull "$VLLM_IMAGE"
    log_success "镜像拉取完成"
}

################################################################################
# 第6步：启动 vLLM 容器
################################################################################
start_vllm_container() {
    print_sep
    log_info "第6步：启动 vLLM GPU 容器 $CONTAINER_NAME"
    print_sep

    if docker ps -a | grep -q "$CONTAINER_NAME"; then
        log_warning "检测到历史容器，先删除"
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi

    docker run -d --name "$CONTAINER_NAME" \
        --gpus all \
        --shm-size=32g \
        -e HF_HOME=/workspace/.cache/huggingface \
        -e http_proxy="$HOST_HTTP_PROXY" \
        -e https_proxy="$HOST_HTTPS_PROXY" \
        -p 8015:8015 \
        -v "$WORKSPACE_DIR":"$WORKSPACE_DIR" \
        "$VLLM_IMAGE" tail -f /dev/null

    sleep 3
    docker ps | grep "$CONTAINER_NAME"
    log_success "容器已启动"
}

################################################################################
# 第7步：容器内安装 Python/vLLM 依赖
################################################################################
configure_vllm_env() {
    print_sep
    log_info "第7步：在容器内安装 Python3 + vLLM 依赖"
    print_sep

    docker exec "$CONTAINER_NAME" bash -c '
set -e
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip python3-venv git curl vim
ln -sf /usr/bin/python3 /usr/bin/python
python3 -m venv /opt/vllm-venv
source /opt/vllm-venv/bin/activate
pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install "vllm>=0.4.2" "transformers>=4.39" accelerate sentencepiece datasets huggingface_hub
pip install "uvicorn>=0.23" fastapi
'

    log_success "容器内依赖安装完成（虚拟环境：/opt/vllm-venv）"
}

################################################################################
# 第8步：生成运行脚本并做烟测
################################################################################
generate_helper_scripts() {
    print_sep
    log_info "第8步：生成 vLLM 运行/压测脚本"
    print_sep

    cat >"$SCRIPT_DIR/run_vllm_server.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

PORT="${VLLM_PORT:-8015}"
MODEL_PATH="${VLLM_MODEL_PATH:-/workspace/models/qwen2.5-32b/qwen2.5-32b}"
CONTAINER="${CONTAINER_NAME:-vllm-gpu}"

docker exec -d "$CONTAINER" bash -c "
source /opt/vllm-venv/bin/activate && \
python -m vllm.entrypoints.openai.api_server \
  --model \"$MODEL_PATH\" \
  --trust-remote-code \
  --tensor-parallel-size \${VLLM_TP:-1} \
  --max-num-seqs \${VLLM_MAX_CONCURRENCY:-6} \
  --port $PORT \
  --host 0.0.0.0 \
  --disable-log-requests \
  --served-model-name \${VLLM_MODEL_NAME:-$(basename "$MODEL_PATH")} \
  > /workspace/logs/vllm_server_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"

echo "vLLM 服务已在容器 $CONTAINER 内启动，端口 $PORT"
EOF

    cat >"$SCRIPT_DIR/bench_vllm_client.sh" <<'EOF'
#!/bin/bash
set -euo pipefail

PORT="${VLLM_PORT:-8015}"
SERVER_HOST="${VLLM_HOST:-127.0.0.1}"
MODEL_NAME="${VLLM_MODEL_NAME:-qwen2.5-32b}"
CONTAINER="${CONTAINER_NAME:-vllm-gpu}"
RESULT_DIR="/workspace/logs/results"
mkdir -p "$RESULT_DIR"

docker exec "$CONTAINER" bash -c "
source /opt/vllm-venv/bin/activate && \
python -m vllm.benchmark.benchmark_serving \
  --url http://${SERVER_HOST}:${PORT}/v1 \
  --model \${VLLM_MODEL_ID:-$MODEL_NAME} \
  --dataset sharegpt \
  --request-rate 5 \
  --num-prompts \${VLLM_NUM_PROMPTS:-200} \
  --max-concurrency \${VLLM_MAX_CONCURRENCY:-6} \
  --save-result \
  --result-dir ${RESULT_DIR} \
  --result-filename benchmark_\$(date +%Y%m%d_%H%M%S).json"
EOF

    chmod +x "$SCRIPT_DIR"/run_vllm_server.sh "$SCRIPT_DIR"/bench_vllm_client.sh
    log_success "脚本已生成：$SCRIPT_DIR/run_vllm_server.sh 等"
}

smoke_test() {
    print_sep
    log_info "附加：容器内打印 vLLM 版本，确认环境"
    print_sep

    docker exec "$CONTAINER_NAME" bash -c 'source /opt/vllm-venv/bin/activate && python -c "import vllm, torch; print(\"vLLM\", vllm.__version__, \"Torch\", torch.__version__)"'
    log_success "vLLM 环境就绪"
}

################################################################################
# 主函数
################################################################################
main() {
    print_sep
    log_info "vLLM 自动部署脚本开始"
    print_sep

    check_environment
    setup_mihomo_proxy
    install_docker
    install_nvidia_toolkit
    prepare_workspace
    pull_vllm_image
    start_vllm_container
    configure_vllm_env
    generate_helper_scripts
    smoke_test

    print_sep
    log_success "🎉 vLLM 环境部署完成！"
    print_sep

    cat <<'EOF'
后续常用命令：
1. 启动服务：    /workspace/scripts/run_vllm_server.sh
2. 压测命令：    /workspace/scripts/bench_vllm_client.sh
3. 进入容器：    docker exec -it vllm-gpu bash
4. 查看日志：    tail -f /workspace/logs/vllm_server_*.log
5. 停止容器：    docker stop vllm-gpu

提示：
- 默认模型路径 /workspace/models/qwen2.5-32b/qwen2.5-32b，按需提前放入。
- 如需代理，请在运行脚本前设置 http_proxy/https_proxy。
- 可通过环境变量覆盖镜像、端口、并发数等参数。
EOF
}

main "$@"

