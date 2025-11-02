# vllm 配置指南 - CUDA 12.8

## 📋 概述

本指南将帮助你将 CUDA 从 12.4 升级到 12.8，并配置 vllm 环境。

## ⚠️ 重要说明：容器内 vs 宿主机

根据你的项目配置，**这些命令需要在容器内执行**：

### 🔍 判断方法

检查你当前的工作环境：

```bash
# 检查是否在容器内
if [ -f /.dockerenv ]; then
    echo "✅ 当前在容器内"
else
    echo "❌ 当前在宿主机上"
fi

# 或者检查容器名称
hostname  # 如果是容器，通常会显示容器ID或名称
```

### 📦 项目中的容器配置

根据项目文档，你使用的是：
- **容器名称**: `cuda-mihomo`
- **基础镜像**: `cuda124-mihomo-python-ubuntu2204:flexflow` (CUDA 12.4)

### 🎯 执行位置

**如果要在容器内使用 vllm，所有命令都应在容器内执行：**

1. **进入容器**：
   ```bash
   docker exec -it cuda-mihomo bash
   ```

2. **在容器内执行所有 CUDA 升级和 vllm 安装命令**

3. **或者使用脚本自动在容器内执行**（见下方）

---

## 🚀 升级步骤

### 前提条件

- Ubuntu 22.04 系统（容器或宿主机）
- 已安装 CUDA 12.4
- root 或 sudo 权限（容器内通常已是 root）
- 网络连接正常

### 步骤 1: 升级 CUDA 到 12.8

**⚠️ 重要：CUDA 升级是系统级操作，不需要激活 conda 环境！**

CUDA 是系统工具，与 Python 环境无关，直接在容器内执行即可。

#### 方法 A: 在容器内执行（推荐）

```bash
# 1. 首先将脚本复制到容器内（从宿主机执行）
docker cp upgrade_cuda_in_container.sh cuda-mihomo:/tmp/

# 2. 进入容器（不需要激活flexflow环境）
docker exec -it cuda-mihomo bash

# 3. 在容器内执行（容器内通常是 root，不需要 sudo）
chmod +x /tmp/upgrade_cuda_in_container.sh
/tmp/upgrade_cuda_in_container.sh
```

#### 方法 B: 直接在容器内执行命令（一行命令）

```bash
# 从宿主机执行，自动进入容器并运行升级
docker exec -it cuda-mihomo bash -c "
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin -O /tmp/cuda-ubuntu2204.pin && \
    mv /tmp/cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    cd /tmp && \
    wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb && \
    cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get -y install cuda-toolkit-12-8
"
```

#### 方法 C: 如果是在宿主机上执行

```bash
# 给脚本添加执行权限
chmod +x upgrade_cuda_to_12.8.sh

# 运行升级脚本（需要 sudo）
sudo ./upgrade_cuda_to_12.8.sh
```

或者手动执行以下命令：

```bash
# 1. 下载并配置 CUDA 仓库 pin 文件
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 2. 下载 CUDA 12.8 安装包
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb

# 3. 安装仓库配置
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb

# 4. 复制 keyring
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

# 5. 更新并安装 CUDA Toolkit
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

### 步骤 2: 配置环境变量

```bash
# 设置 CUDA 12.8 环境变量
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**永久配置（推荐）:**

创建 `/etc/profile.d/cuda-12.8.sh` 文件：

```bash
sudo tee /etc/profile.d/cuda-12.8.sh << 'EOF'
# CUDA 12.8 环境变量配置
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

# 使配置立即生效
source /etc/profile.d/cuda-12.8.sh
```

### 步骤 3: 验证 CUDA 安装

```bash
# 检查 CUDA 版本
nvcc --version

# 应该显示 CUDA 12.8

# 检查 GPU（如果驱动已安装）
nvidia-smi
```

### 步骤 4: 安装 uv（Python 包管理器）

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 添加到 PATH（如果还没添加）
export PATH="$HOME/.cargo/bin:$PATH"

# 验证安装
uv --version
```

### 步骤 5: 创建 Python 环境并安装 vllm

**⚠️ 重要：这里需要初始化 conda，但不需要激活 flexflow 环境！**

**原因**：
- flexflow 环境是 Python 3.10.18，而 vllm 需要 Python 3.12
- 建议创建新的专用 vllm 环境，避免冲突

**在容器内执行以下命令：**

```bash
# 进入容器（如果还没有进入）
docker exec -it cuda-mihomo bash

# 初始化 conda（必须先执行这一步才能使用 conda 命令）
source /opt/miniforge3/etc/profile.d/conda.sh

# 创建新的 Python 3.12 环境（不要使用 flexflow 环境）
conda create -n vllm python=3.12 -y

# 激活新创建的 vllm 环境
conda activate vllm

# 使用 uv 安装 vllm（指定 CUDA 12.8 后端）
# 先安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# 安装 vllm
uv pip install vllm --torch-backend=cu128
```

**或者使用 pip 直接安装:**

```bash
# 激活 conda 环境
conda activate vllm

# 安装 vllm（需要 PyTorch CUDA 12.8 支持）
pip install vllm

# 如果遇到 torch 版本问题，可以指定 torch 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install vllm
```

### 步骤 6: 验证 vllm 安装

```bash
# 激活环境
conda activate vllm

# 测试导入
python -c "import vllm; print(vllm.__version__)"

# 检查 CUDA 支持
python -c "from vllm import LLM; print('vllm 安装成功，CUDA 支持正常')"
```

## 🔍 故障排除

### 问题 1: nvcc 命令未找到

**解决方案:**
```bash
# 检查 CUDA 安装路径
ls -la /usr/local/cuda-12.8/bin/nvcc

# 手动设置环境变量
export PATH=/usr/local/cuda-12.8/bin:$PATH
```

### 问题 2: vllm 安装失败

**可能原因:**
- PyTorch 版本不兼容
- CUDA 版本不匹配

**解决方案:**
```bash
# 先安装兼容的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 然后再安装 vllm
pip install vllm
```

### 问题 3: CUDA 驱动版本问题

**检查驱动版本:**
```bash
nvidia-smi
```

**要求:** NVIDIA 驱动版本 >= 570.86.10 (与 CUDA 12.8 匹配)

如果驱动版本过低，需要升级 NVIDIA 驱动。

## 📚 相关资源

- [CUDA 12.8 官方文档](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [vllm 官方文档](https://docs.vllm.ai/)
- [PyTorch CUDA 12.8 安装](https://pytorch.org/get-started/locally/)

## 🐳 容器内快速执行（一键脚本）

我已经创建了专门用于容器内执行的脚本 `upgrade_cuda_in_container.sh`：

### 方法 1: 从宿主机一键执行（推荐）

```bash
# 从宿主机执行，自动在容器内升级 CUDA
docker exec -it cuda-mihomo bash < upgrade_cuda_in_container.sh
```

### 方法 2: 复制脚本到容器内执行

```bash
# 1. 复制脚本到容器
docker cp upgrade_cuda_in_container.sh cuda-mihomo:/tmp/

# 2. 在容器内执行
docker exec -it cuda-mihomo bash /tmp/upgrade_cuda_in_container.sh
```

### 方法 3: 进入容器后手动执行

```bash
# 1. 进入容器
docker exec -it cuda-mihomo bash

# 2. 在容器内执行脚本（如果已复制到容器）
bash /tmp/upgrade_cuda_in_container.sh

# 或者直接在容器内执行命令（见上方步骤 1）
```

## ⚠️ 注意事项

1. **执行位置**: **这些命令需要在容器内执行**（如果使用 Docker 容器）
2. **备份重要数据**: 升级 CUDA 前建议备份重要配置文件
3. **兼容性检查**: 确保你的 GPU 支持 CUDA 12.8
4. **驱动版本**: 确保宿主机的 NVIDIA 驱动版本 >= 570.86.10（容器通过 `--gpus all` 使用宿主机驱动）
5. **环境隔离**: 建议使用 conda 环境来隔离不同项目的依赖
6. **容器持久化**: 容器内的更改在容器删除后会丢失，如需持久化，建议：
   - 重新构建包含 CUDA 12.8 的镜像
   - 或者使用数据卷保存配置

## 🔄 回退到 CUDA 12.4

如果遇到问题需要回退：

```bash
# 卸载 CUDA 12.8
sudo apt-get remove --purge cuda-toolkit-12-8

# 重新安装 CUDA 12.4（如果需要）
# 参考之前的 CUDA 12.4 安装步骤
```

