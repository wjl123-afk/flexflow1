#!/bin/bash
set -e

################################################################################
# FlexFlow Llama-2-7b 缓存恢复脚本
# 功能：
#   1. 检查 HuggingFace 缓存是否存在且完整
#   2. 如果不完整，从 tar 文件恢复
#   3. 自动构建正确的目录结构
#
# 使用方法：
#   bash restore_llama2_from_tar.sh [TAR_FILE]
#
# 参数：
#   TAR_FILE  - Llama-2-7b tar 文件路径（可选）
#               默认: /workspace/archive_old_files/Llama-2-7b.tar
#
# 示例：
#   bash restore_llama2_from_tar.sh
#   bash restore_llama2_from_tar.sh /path/to/custom/Llama-2-7b.tar
#   bash restore_llama2_from_tar.sh /mnt/sftp/models/Llama-2-7b.tar
################################################################################

# 显示使用说明
show_usage() {
    echo "使用方法: $0 [TAR_FILE]"
    echo ""
    echo "参数:"
    echo "  TAR_FILE    Llama-2-7b tar 文件路径（可选）"
    echo "              默认: /workspace/archive_old_files/Llama-2-7b.tar"
    echo ""
    echo "示例:"
    echo "  $0"
    echo "  $0 /path/to/Llama-2-7b.tar"
    echo "  $0 /mnt/sftp/models/Llama-2-7b.tar"
    echo ""
    exit 1
}

# 解析参数
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
fi

# 配置变量
DEFAULT_TAR_FILE="/workspace/archive_old_files/Llama-2-7b.tar"
TAR_FILE="${1:-$DEFAULT_TAR_FILE}"  # 使用第一个参数，如果没有则使用默认值
COMMIT_HASH="01c7f73d771dfac7d292323805ebc428287df4f9"
HF_CACHE_BASE="/root/.cache/huggingface/hub/models--meta-llama--llama-2-7b-hf"
SNAPSHOT_DIR="${HF_CACHE_BASE}/snapshots/${COMMIT_HASH}"
TMP_RESTORE_DIR="/tmp/llama2_restore"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "🦙 FlexFlow Llama-2-7b 缓存恢复脚本"
echo "======================================================================"
echo ""
echo "📦 tar 文件: ${TAR_FILE}"
if [ "${TAR_FILE}" != "${DEFAULT_TAR_FILE}" ]; then
    echo "   (自定义路径)"
else
    echo "   (默认路径)"
fi
echo ""

################################################################################
# 函数：检查缓存完整性
################################################################################
check_cache_integrity() {
    echo "🔍 检查缓存完整性..."
    echo ""
    
    local is_complete=true
    
    # 1. 检查基础目录
    if [ ! -d "${HF_CACHE_BASE}" ]; then
        echo -e "${YELLOW}  ⚠️  基础目录不存在: ${HF_CACHE_BASE}${NC}"
        is_complete=false
    fi
    
    # 2. 检查 refs/main
    if [ ! -f "${HF_CACHE_BASE}/refs/main" ]; then
        echo -e "${YELLOW}  ⚠️  refs/main 不存在${NC}"
        is_complete=false
    else
        local ref_content=$(cat "${HF_CACHE_BASE}/refs/main")
        if [ "${ref_content}" != "${COMMIT_HASH}" ]; then
            echo -e "${YELLOW}  ⚠️  refs/main 内容不匹配 (期望: ${COMMIT_HASH}, 实际: ${ref_content})${NC}"
            is_complete=false
        else
            echo -e "${GREEN}  ✅ refs/main 正确${NC}"
        fi
    fi
    
    # 3. 检查 snapshot 目录
    if [ ! -d "${SNAPSHOT_DIR}" ]; then
        echo -e "${YELLOW}  ⚠️  snapshot 目录不存在: ${SNAPSHOT_DIR}${NC}"
        is_complete=false
    fi
    
    # 4. 检查关键文件
    local key_files=(
        "config.json"
        "tokenizer.model"
        "tokenizer_config.json"
    )
    
    for file in "${key_files[@]}"; do
        if [ ! -f "${SNAPSHOT_DIR}/${file}" ]; then
            echo -e "${YELLOW}  ⚠️  关键文件缺失: ${file}${NC}"
            is_complete=false
        else
            echo -e "${GREEN}  ✅ ${file} 存在${NC}"
        fi
    done
    
    # 5. 检查 safetensors 文件
    local safetensors_count=$(find "${SNAPSHOT_DIR}" -name "*.safetensors" 2>/dev/null | wc -l)
    if [ "${safetensors_count}" -lt 3 ]; then
        echo -e "${YELLOW}  ⚠️  safetensors 文件不完整 (期望: 3, 实际: ${safetensors_count})${NC}"
        is_complete=false
    else
        echo -e "${GREEN}  ✅ safetensors 文件完整 (${safetensors_count} 个)${NC}"
    fi
    
    # 6. 检查 half-precision 目录（可选）
    if [ -d "${SNAPSHOT_DIR}/half-precision" ]; then
        local weight_count=$(find "${SNAPSHOT_DIR}/half-precision" -name "*.weight" 2>/dev/null | wc -l)
        if [ "${weight_count}" -gt 100 ]; then
            echo -e "${GREEN}  ✅ half-precision 权重已存在 (${weight_count} 个文件)${NC}"
            echo -e "${GREEN}  💡 权重已转换，可直接使用${NC}"
        else
            echo -e "${YELLOW}  ⚠️  half-precision 目录不完整 (${weight_count} 个文件)${NC}"
        fi
    else
        echo -e "${BLUE}  ℹ️  half-precision 目录不存在（首次运行时会自动创建）${NC}"
    fi
    
    echo ""
    
    if [ "${is_complete}" = true ]; then
        return 0  # 完整
    else
        return 1  # 不完整
    fi
}

################################################################################
# 函数：恢复缓存
################################################################################
restore_cache() {
    echo "======================================================================"
    echo "🔧 开始恢复缓存..."
    echo "======================================================================"
    echo ""
    
    # Step 1: 检查 tar 文件
    echo "📦 Step 1: 检查 tar 文件..."
    if [ ! -f "${TAR_FILE}" ]; then
        echo -e "${RED}❌ 错误: tar 文件不存在: ${TAR_FILE}${NC}"
        exit 1
    fi
    local tar_size=$(du -sh "${TAR_FILE}" | awk '{print $1}')
    echo -e "${GREEN}✅ tar 文件存在 (${tar_size})${NC}"
    echo ""
    
    # Step 2: 清理现有缓存（如果存在）
    echo "🗑️  Step 2: 清理现有缓存..."
    if [ -d "${HF_CACHE_BASE}" ]; then
        echo "   备份现有缓存..."
        local backup_name="${HF_CACHE_BASE}.backup.$(date +%Y%m%d_%H%M%S)"
        mv "${HF_CACHE_BASE}" "${backup_name}"
        echo -e "${GREEN}   ✅ 已备份到: ${backup_name}${NC}"
    else
        echo "   (无需清理)"
    fi
    echo ""
    
    # Step 3: 解压 tar 文件
    echo "📂 Step 3: 解压 tar 文件..."
    rm -rf "${TMP_RESTORE_DIR}"
    mkdir -p "${TMP_RESTORE_DIR}"
    echo "   开始解压（需要 2-3 分钟）..."
    tar -xf "${TAR_FILE}" -C "${TMP_RESTORE_DIR}"
    echo -e "${GREEN}✅ 解压完成${NC}"
    echo ""
    
    # Step 4: 查找模型文件位置
    echo "🔍 Step 4: 查找模型文件..."
    local config_path=$(find "${TMP_RESTORE_DIR}" -name "config.json" -type f | head -1)
    if [ -z "${config_path}" ]; then
        echo -e "${RED}❌ 错误: 未找到 config.json${NC}"
        exit 1
    fi
    local model_source=$(dirname "${config_path}")
    echo -e "${GREEN}✅ 模型文件位置: ${model_source}${NC}"
    echo ""
    
    # Step 5: 创建 HF 缓存结构
    echo "📁 Step 5: 创建 HF 缓存结构..."
    mkdir -p "${HF_CACHE_BASE}/refs"
    mkdir -p "${SNAPSHOT_DIR}"
    echo "${COMMIT_HASH}" > "${HF_CACHE_BASE}/refs/main"
    echo -e "${GREEN}✅ 目录结构已创建${NC}"
    echo "   refs/main: $(cat ${HF_CACHE_BASE}/refs/main)"
    echo ""
    
    # Step 6: 复制模型文件
    echo "📦 Step 6: 复制模型文件..."
    echo "   (需要 1-2 分钟)..."
    cp -r "${model_source}"/* "${SNAPSHOT_DIR}/"
    echo -e "${GREEN}✅ 文件已复制${NC}"
    echo ""
    
    # Step 7: 验证关键文件
    echo "🔍 Step 7: 验证关键文件..."
    local files_ok=true
    
    # 验证 config.json
    if [ -f "${SNAPSHOT_DIR}/config.json" ]; then
        echo -e "${GREEN}  ✅ config.json ($(du -sh ${SNAPSHOT_DIR}/config.json | awk '{print $1}'))${NC}"
    else
        echo -e "${RED}  ❌ config.json 缺失${NC}"
        files_ok=false
    fi
    
    # 验证 tokenizer.model
    if [ -f "${SNAPSHOT_DIR}/tokenizer.model" ]; then
        echo -e "${GREEN}  ✅ tokenizer.model ($(du -sh ${SNAPSHOT_DIR}/tokenizer.model | awk '{print $1}'))${NC}"
    else
        echo -e "${RED}  ❌ tokenizer.model 缺失${NC}"
        files_ok=false
    fi
    
    # 验证 safetensors
    local safetensors_files=$(ls -1 "${SNAPSHOT_DIR}"/*.safetensors 2>/dev/null | wc -l)
    if [ "${safetensors_files}" -ge 3 ]; then
        echo -e "${GREEN}  ✅ safetensors 文件 (${safetensors_files} 个)${NC}"
        ls -lh "${SNAPSHOT_DIR}"/*.safetensors | awk '{print "     -", $9, "(" $5 ")"}'
    else
        echo -e "${RED}  ❌ safetensors 文件不完整 (${safetensors_files} 个)${NC}"
        files_ok=false
    fi
    
    echo ""
    
    if [ "${files_ok}" = false ]; then
        echo -e "${RED}❌ 验证失败: 关键文件不完整${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ 所有关键文件验证通过${NC}"
    echo ""
    
    # Step 8: 清理临时文件
    echo "🗑️  Step 8: 清理临时文件..."
    rm -rf "${TMP_RESTORE_DIR}"
    echo -e "${GREEN}✅ 临时文件已清理${NC}"
    echo ""
    
    # 显示最终状态
    echo "📊 最终状态:"
    echo "   缓存目录: ${HF_CACHE_BASE}"
    echo "   总大小: $(du -sh ${HF_CACHE_BASE} | awk '{print $1}')"
    echo ""
}

################################################################################
# 主流程
################################################################################

# 检查完整性
if check_cache_integrity; then
    echo "======================================================================"
    echo -e "${GREEN}✅ 缓存完整，无需恢复${NC}"
    echo "======================================================================"
    echo ""
    echo "📊 缓存信息:"
    echo "   位置: ${HF_CACHE_BASE}"
    echo "   大小: $(du -sh ${HF_CACHE_BASE} | awk '{print $1}')"
    
    # 检查是否有转换后的权重
    if [ -d "${SNAPSHOT_DIR}/half-precision" ]; then
        weight_count=$(find "${SNAPSHOT_DIR}/half-precision" -name "*.weight" 2>/dev/null | wc -l)
        echo "   权重: 已转换 (${weight_count} 个 .weight 文件)"
    else
        echo "   权重: 未转换（首次运行时会自动转换，需要 5-10 分钟）"
    fi
    
    echo ""
    echo "🚀 可以直接运行测试:"
    echo "   python /workspace/test_llama2_from_tar.py -ll:gpu 1 -ll:zsize 40000 -ll:fsize 14000 -ll:cpu 4"
    echo ""
    exit 0
else
    echo "======================================================================"
    echo -e "${YELLOW}⚠️  缓存不完整，需要恢复${NC}"
    echo "======================================================================"
    echo ""
    
    # 执行恢复
    restore_cache
    
    echo "======================================================================"
    echo -e "${GREEN}🎉 缓存恢复完成！${NC}"
    echo "======================================================================"
    echo ""
    echo "📝 下一步:"
    echo "   1. 运行测试脚本（首次会自动转换权重，需要 5-10 分钟）:"
    echo "      cd /workspace/flexflow-serve/examples"
    echo "      source /opt/miniforge3/etc/profile.d/conda.sh"
    echo "      conda activate flexflow"
    echo "      export BUILD_FOLDER=/workspace/flexflow-serve/build"
    echo "      export PYTHON_FOLDER=/workspace/flexflow-serve/python"
    echo "      export PYLIB_PATH=\$(python \$PYTHON_FOLDER/flexflow/findpylib.py)"
    echo "      export PYLIB_DIR=\$(dirname \$PYLIB_PATH)"
    echo "      export LD_LIBRARY_PATH=\"\$BUILD_FOLDER:\$BUILD_FOLDER/deps/legion/lib:\$PYLIB_DIR:\$LD_LIBRARY_PATH\""
    echo "      export PYTHONPATH=\"\$PYTHON_FOLDER:\$BUILD_FOLDER/deps/legion/bindings/python:\$PYTHONPATH\""
    echo "      python -u /workspace/test_llama2_from_tar.py -ll:gpu 1 -ll:zsize 40000 -ll:fsize 14000 -ll:cpu 4"
    echo ""
    echo "   2. 第二次运行会快很多（直接加载转换后的权重）"
    echo ""
fi

