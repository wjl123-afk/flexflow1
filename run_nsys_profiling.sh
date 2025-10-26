#!/bin/bash
# Nsight Systems Profiling for Layer & Operator Analysis
# 使用Nsight Systems获取GPU kernel级别的性能数据

set -e

echo "======================================================================"
echo "🔬 Nsight Systems Layer & Operator Profiling"
echo "======================================================================"
echo ""

# 配置
WORKSPACE="/workspace"
TEST_SCRIPT="$WORKSPACE/test_parallel_strategies.py"
RESULT_DIR="$WORKSPACE/nsys_profiling_results"
LOG_DIR="$RESULT_DIR/logs"

# 创建输出目录
mkdir -p "$RESULT_DIR"
mkdir -p "$LOG_DIR"

echo "📁 结果目录: $RESULT_DIR"
echo ""

# 测试配置
# 格式: "策略编号:描述:TP:PP:GPU数"
strategies=(
    "1:TP1_PP1:1:1:1"
    "4:TP2_PP1:2:1:2"
    "6:TP4_PP1:4:1:4"
)

seq_length=128

echo "📋 测试策略:"
for strategy in "${strategies[@]}"; do
    IFS=':' read -r idx desc tp pp num_gpus <<< "$strategy"
    echo "  $idx. $desc (TP=$tp, PP=$pp, GPU=$num_gpus)"
done
echo ""

# 检查nsys是否可用
if ! command -v nsys &> /dev/null; then
    echo "❌ 错误: nsys 未安装"
    echo "请安装 NVIDIA Nsight Systems:"
    echo "  https://developer.nvidia.com/nsight-systems"
    exit 1
fi

echo "✅ Nsight Systems 已安装"
nsys --version
echo ""

# 运行profiling
for strategy in "${strategies[@]}"; do
    IFS=':' read -r idx desc tp pp num_gpus <<< "$strategy"
    
    echo "======================================================================"
    echo "🔬 Profiling: Strategy $idx - $desc"
    echo "======================================================================"
    
    output_file="$RESULT_DIR/llama2_${desc}_sl${seq_length}"
    log_file="$LOG_DIR/nsys_${desc}_sl${seq_length}.log"
    
    echo "📝 输出文件: ${output_file}.nsys-rep"
    echo "📝 日志文件: $log_file"
    echo ""
    
    # 运行nsys profiling
    # --trace: 指定要追踪的API
    # --cuda-memory-usage: 追踪CUDA内存使用
    # --force-overwrite: 覆盖已有文件
    # --show-output: 显示应用程序输出
    
    echo "⏳ 开始profiling（这可能需要5-10分钟）..."
    
    export RESULT_DIR="$RESULT_DIR"
    
    if nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --show-output=true \
        --output="$output_file" \
        python -u "$TEST_SCRIPT" "$idx" "$seq_length" \
            -ll:gpu "$num_gpus" \
            -ll:zsize $((20000 * num_gpus)) \
            -ll:fsize 14000 \
            -ll:cpu $((4 * num_gpus)) \
        2>&1 | tee "$log_file"; then
        
        echo "✅ Profiling 完成: $desc"
        
        # 生成文本报告
        report_file="${output_file}_report.txt"
        echo "📊 生成文本报告: $report_file"
        
        nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
            --format table \
            "${output_file}.nsys-rep" > "$report_file" 2>&1 || true
        
        echo "✅ 报告生成完成"
        
    else
        echo "❌ Profiling 失败: $desc"
    fi
    
    echo ""
    unset RESULT_DIR
done

echo "======================================================================"
echo "✅ 所有profiling完成！"
echo "======================================================================"
echo ""
echo "📁 结果位置: $RESULT_DIR"
echo ""
echo "📊 生成的文件:"
ls -lh "$RESULT_DIR"/*.nsys-rep 2>/dev/null || echo "  (未找到.nsys-rep文件)"
echo ""
echo "📝 查看文本报告:"
echo "  cat $RESULT_DIR/*_report.txt"
echo ""
echo "🖥️  使用GUI分析（需要在本地机器上）:"
echo "  1. 下载.nsys-rep文件到本地"
echo "  2. 使用Nsight Systems GUI打开:"
echo "     nsys-ui llama2_TP*_sl128.nsys-rep"
echo ""
echo "🔍 分析重点:"
echo "  - GPU Kernel执行时间分布"
echo "  - 计算密集型 vs 访存密集型kernel"
echo "  - TP并行效率（对比TP1 vs TP2 vs TP4）"
echo "  - 通信kernel（NCCL AllReduce）占比"
echo ""

