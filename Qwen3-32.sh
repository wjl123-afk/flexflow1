#!/bin/bash

# ==========================================
# Qwen3-32B 模型性能测试 - 一键执行脚本
# ==========================================
# 功能：
#   1. 自动启动 vllm serve 服务
#   2. 等待服务就绪
#   3. 下载 BurstGPT 数据集（可选）
#   4. 运行性能测试（输入2048/输出2048 tokens）
#   5. 保存测试结果
# ==========================================

set -e

echo "========================================="
echo "🚀 Qwen3-32B 模型性能测试"
echo "配置: 输入 2048 / 输出 2048 tokens"
echo "========================================="
echo ""

# ==========================================
# 配置参数（可根据需要修改）
# ==========================================
MODEL_PATH="/workspace/models/Qwen/Qwen3-32B"
MODEL_NAME="Qwen3-32B"
RESULT_DIR="/workspace/results"
LOG_DIR="/workspace"

# 测试参数
INPUT_LEN=2048
OUTPUT_LEN=2048
NUM_PROMPTS=512
MAX_CONCURRENCY=32

# 服务参数
GPU_MEMORY_UTIL=0.80
MAX_MODEL_LEN=2048
TENSOR_PARALLEL_SIZE=4
MAX_NUM_SEQS=128

# 数据集选择：random 或 burstgpt
DATASET_NAME="random"  # 或 "burstgpt"

# ==========================================
# 步骤 0: 环境检查和准备
# ==========================================
echo "📦 步骤 0: 环境检查和准备..."

# 检查 Conda
if [ ! -f "/opt/miniforge3/etc/profile.d/conda.sh" ]; then
    echo "❌ Conda 初始化脚本不存在"
    exit 1
fi

source /opt/miniforge3/etc/profile.d/conda.sh

# 检查并激活 vllm 环境
if ! conda env list | grep -q "^vllm"; then
    echo "❌ vllm conda 环境不存在"
    echo "请先创建: conda create -n vllm python=3.12 -y"
    exit 1
fi

conda activate vllm || {
    echo "❌ 无法激活 vllm 环境"
    exit 1
}

# 检查 vLLM 是否安装
if ! python -c "import vllm" 2>/dev/null; then
    echo "❌ vLLM 未安装"
    echo "请安装: pip install vllm"
    exit 1
fi

# 设置 CUDA 环境变量
if [ -f "/etc/profile.d/cuda-12.8.sh" ]; then
    source /etc/profile.d/cuda-12.8.sh
    echo "✅ 已加载 CUDA 12.8 环境变量"
elif [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "✅ 已自动设置 CUDA 12.8 环境变量"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "✅ 已自动设置 CUDA 环境变量"
else
    echo "⚠️  未找到 CUDA 路径，可能通过容器环境提供"
fi

# 设置 PyTorch CUDA 分配配置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "✅ 已设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# 验证 PyTorch CUDA（延迟检查，确保环境变量生效）
sleep 1
if python -c "import torch; print('OK' if torch.cuda.is_available() else 'FAIL')" 2>/dev/null | grep -q "FAIL"; then
    echo "⚠️  PyTorch CUDA 可能不可用"
    echo "   如果后续测试失败，请检查:"
    echo "   1. 容器是否使用 --gpus all 启动"
    echo "   2. PyTorch 是否安装了 CUDA 版本"
    echo "   3. 运行: python -c 'import torch; print(torch.cuda.is_available())'"
else
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    echo "✅ PyTorch CUDA 可用（GPU 数量: $GPU_COUNT）"
fi

echo "✅ 环境检查和准备完成"
echo ""

# ==========================================
# 检查模型路径
# ==========================================
echo "🔍 检查模型路径..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 模型路径不存在: $MODEL_PATH"
    echo "请检查模型是否已下载"
    exit 1
fi
echo "✅ 模型路径: $MODEL_PATH"
echo ""

# ==========================================
# 创建输出目录
# ==========================================
mkdir -p "$RESULT_DIR"
mkdir -p "$LOG_DIR"

# ==========================================
# 步骤 1: 下载数据集（如果需要）
# ==========================================
if [ "$DATASET_NAME" == "burstgpt" ]; then
    echo "📥 步骤 1: 下载 BurstGPT 数据集..."
    
    DATASET_FILE="$LOG_DIR/BurstGPT_without_fails_2.csv"
    
    if [ ! -f "$DATASET_FILE" ]; then
        export http_proxy=http://172.17.0.1:7890
        export https_proxy=http://172.17.0.1:7890
        
        echo "开始下载..."
        if wget -q --timeout=30 --tries=3 \
          https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv \
          -O "$DATASET_FILE"; then
            SIZE=$(du -sh "$DATASET_FILE" | awk '{print $1}')
            echo "✅ 数据集下载完成，大小: $SIZE"
        else
            echo "⚠️  数据集下载失败，将使用随机数据集"
            DATASET_NAME="random"
        fi
    else
        SIZE=$(du -sh "$DATASET_FILE" | awk '{print $1}')
        echo "✅ 数据集已存在，大小: $SIZE"
    fi
    echo ""
fi

# ==========================================
# 步骤 2: 停止旧服务
# ==========================================
echo "🛑 步骤 2: 检查并停止旧服务..."
pkill -f "vllm serve" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 3
echo "✅ 旧服务已停止"
echo ""

# ==========================================
# 步骤 3: 启动 vllm serve 服务
# ==========================================
echo "🚀 步骤 3: 启动 vllm serve 服务..."
echo "  模型: $MODEL_PATH"
echo "  GPU显存利用率: ${GPU_MEMORY_UTIL}"
echo "  最大序列长度: ${MAX_MODEL_LEN}"
echo "  张量并行: ${TENSOR_PARALLEL_SIZE}"
echo ""

SERVER_LOG="$LOG_DIR/vllm_server.log"
nohup vllm serve \
  "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --dtype float16 \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "服务进程 PID: $SERVER_PID"
echo "日志文件: $SERVER_LOG"
echo ""

# ==========================================
# 步骤 4: 等待服务就绪
# ==========================================
echo "⏳ 步骤 4: 等待服务启动（最多 180 秒）..."
MAX_WAIT=180
WAIT_INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ 服务已启动（等待了 ${ELAPSED} 秒）"
        break
    fi
    
    # 检查进程是否还在运行
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo ""
        echo "❌ 服务进程意外退出！"
        echo "查看日志:"
        tail -50 "$SERVER_LOG"
        exit 1
    fi
    
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    if [ $((ELAPSED % 15)) -eq 0 ]; then
        echo "  ... 已等待 ${ELAPSED}/${MAX_WAIT} 秒"
    fi
    sleep $WAIT_INTERVAL
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "❌ 服务启动超时（超过 ${MAX_WAIT} 秒）"
    echo "查看日志:"
    tail -50 "$SERVER_LOG"
    echo ""
    echo "尝试手动检查:"
    echo "  curl http://localhost:8000/health"
    echo "  tail -f $SERVER_LOG"
    exit 1
fi

echo ""

# ==========================================
# 步骤 5: 运行性能测试
# ==========================================
echo "📊 步骤 5: 运行性能测试..."
echo "  数据集: $DATASET_NAME"
echo "  输入长度: ${INPUT_LEN} tokens"
echo "  输出长度: ${OUTPUT_LEN} tokens"
echo "  测试样本数: ${NUM_PROMPTS}"
echo "  最大并发: ${MAX_CONCURRENCY}"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="$RESULT_DIR/Qwen3-32B_${DATASET_NAME}_${INPUT_LEN}_${OUTPUT_LEN}_${TIMESTAMP}.json"
BENCH_LOG="$LOG_DIR/benchmark_${TIMESTAMP}.log"

# 构建 benchmark 命令
BENCH_CMD="vllm bench serve \
  --save-result \
  --save-detailed \
  --ignore-eos \
  --backend vllm \
  --model $MODEL_PATH \
  --endpoint /v1/completions \
  --num-prompts $NUM_PROMPTS \
  --request-rate inf \
  --max-concurrency $MAX_CONCURRENCY \
  --metric-percentile 50,90,99 \
  --percentile-metrics ttft,tpot,itl \
  --result-dir $RESULT_DIR \
  --result-filename $(basename $RESULT_FILE)"

# 根据数据集类型添加参数
if [ "$DATASET_NAME" == "burstgpt" ] && [ -f "$LOG_DIR/BurstGPT_without_fails_2.csv" ]; then
    BENCH_CMD="$BENCH_CMD --dataset-name burstgpt --dataset-path $LOG_DIR/BurstGPT_without_fails_2.csv"
else
    BENCH_CMD="$BENCH_CMD --dataset-name random --random-input-len $INPUT_LEN --random-output-len $OUTPUT_LEN --random-range-ratio 0"
fi

echo "执行命令:"
echo "$BENCH_CMD"
echo ""

# 运行测试
if eval "$BENCH_CMD" 2>&1 | tee "$BENCH_LOG"; then
    echo ""
    echo "✅ 性能测试完成"
else
    TEST_EXIT_CODE=$?
    echo ""
    echo "⚠️  测试过程中可能出现错误（退出码: $TEST_EXIT_CODE）"
    echo "查看日志: $BENCH_LOG"
fi

echo ""

# ==========================================
# 步骤 6: 显示结果摘要
# ==========================================
echo "========================================="
echo "📋 测试结果摘要"
echo "========================================="
echo ""

if [ -f "$RESULT_FILE" ]; then
    echo "✅ 结果文件: $RESULT_FILE"
    
    # 尝试提取关键指标（如果 JSON 格式正确）
    if command -v python3 > /dev/null 2>&1; then
        python3 << EOF
import json
import sys

try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)
    
    print("\n=========================================")
    print("📊 关键性能指标")
    print("=========================================")
    
    # 请求统计
    print("\n【请求统计】")
    if 'successful_requests' in data:
        print(f"  成功请求数: {data['successful_requests']}")
    if 'failed_requests' in data:
        print(f"  失败请求数: {data.get('failed_requests', 0)}")
    if 'benchmark_duration' in data:
        print(f"  测试时长: {data['benchmark_duration']:.2f} 秒")
    
    # Token 统计
    print("\n【Token 统计】")
    if 'total_input_tokens' in data:
        print(f"  总输入 tokens: {data['total_input_tokens']:,}")
    if 'total_generated_tokens' in data or 'total_output_tokens' in data:
        tokens = data.get('total_generated_tokens') or data.get('total_output_tokens', 0)
        print(f"  总生成 tokens: {tokens:,}")
    if 'total_tokens' in data:
        print(f"  总 tokens: {data['total_tokens']:,}")
    
    # 吞吐量指标
    print("\n【吞吐量指标】")
    if 'request_throughput' in data:
        print(f"  请求吞吐量: {data['request_throughput']:.2f} req/s")
    if 'output_token_throughput' in data:
        print(f"  输出 token 吞吐量: {data['output_token_throughput']:.2f} tok/s")
    if 'peak_output_token_throughput' in data:
        print(f"  峰值输出 token 吞吐量: {data['peak_output_token_throughput']:.2f} tok/s")
    if 'total_token_throughput' in data:
        print(f"  总 token 吞吐量: {data['total_token_throughput']:.2f} tok/s")
    if 'peak_concurrent_requests' in data:
        print(f"  峰值并发请求数: {data['peak_concurrent_requests']:.2f}")
    
    # TTFT 指标
    print("\n【Time to First Token (TTFT)】")
    if 'mean_ttft' in data:
        print(f"  平均 TTFT: {data['mean_ttft']:.2f} ms")
    if 'median_ttft' in data:
        print(f"  中位数 TTFT: {data['median_ttft']:.2f} ms")
    if 'p50_ttft' in data:
        print(f"  P50 TTFT: {data['p50_ttft']:.2f} ms")
    if 'p90_ttft' in data:
        print(f"  P90 TTFT: {data['p90_ttft']:.2f} ms")
    if 'p99_ttft' in data:
        print(f"  P99 TTFT: {data['p99_ttft']:.2f} ms")
    
    # TPOT 指标
    print("\n【Time per Output Token (TPOT)】")
    if 'mean_tpot' in data:
        print(f"  平均 TPOT: {data['mean_tpot']:.2f} ms")
    if 'median_tpot' in data:
        print(f"  中位数 TPOT: {data['median_tpot']:.2f} ms")
    if 'p50_tpot' in data:
        print(f"  P50 TPOT: {data['p50_tpot']:.2f} ms")
    if 'p90_tpot' in data:
        print(f"  P90 TPOT: {data['p90_tpot']:.2f} ms")
    if 'p99_tpot' in data:
        print(f"  P99 TPOT: {data['p99_tpot']:.2f} ms")
    
    # ITL 指标（如果有）
    if 'mean_itl' in data or 'median_itl' in data:
        print("\n【Inter-token Latency (ITL)】")
        if 'mean_itl' in data:
            print(f"  平均 ITL: {data['mean_itl']:.2f} ms")
        if 'median_itl' in data:
            print(f"  中位数 ITL: {data['median_itl']:.2f} ms")
        if 'p50_itl' in data:
            print(f"  P50 ITL: {data['p50_itl']:.2f} ms")
        if 'p90_itl' in data:
            print(f"  P90 ITL: {data['p90_itl']:.2f} ms")
        if 'p99_itl' in data:
            print(f"  P99 ITL: {data['p99_itl']:.2f} ms")
    
    print("\n=========================================")
    print("完整结果请查看: $RESULT_FILE")
    print("=========================================")
    
except Exception as e:
    print(f"\n⚠️  无法解析结果文件: {e}")
    import traceback
    traceback.print_exc()
    print("\n请手动查看: $RESULT_FILE")
EOF
    fi
else
    echo "⚠️  结果文件未找到: $RESULT_FILE"
    echo "请检查测试是否成功完成"
fi

echo ""
echo "========================================="
echo "📝 日志文件"
echo "========================================="
echo "  服务日志: $SERVER_LOG"
echo "  测试日志: $BENCH_LOG"
echo ""

# ==========================================
# 步骤 7: 可选 - 停止服务
# ==========================================
read -p "是否停止 vllm serve 服务？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "停止服务..."
    kill $SERVER_PID 2>/dev/null || true
    sleep 2
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
    echo "✅ 服务已停止"
else
    echo "ℹ️  服务将继续运行（PID: $SERVER_PID）"
    echo "   如需停止，运行: kill $SERVER_PID"
fi

echo ""
echo "========================================="
echo "🎉 测试流程完成！"
echo "========================================="
echo ""

