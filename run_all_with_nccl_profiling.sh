#!/bin/bash
# FlexFlow并行策略测试 - 带NCCL通信profiling
# 为所有策略收集真实的通信数据

set -e

echo "======================================================================"
echo "🚀 FlexFlow 并行策略性能测试套件 (含NCCL Profiling)"
echo "======================================================================"
echo ""

# 配置变量
WORKSPACE="/workspace"
TEST_SCRIPT="$WORKSPACE/test_parallel_strategies.py"
RESULT_DIR="$WORKSPACE/test_results_with_nccl"  # 新目录，不覆盖旧结果
NCCL_LOG_DIR="$RESULT_DIR/nccl_logs"
LOG_DIR="$RESULT_DIR/profiling_logs"

# 创建目录
mkdir -p "$RESULT_DIR"
mkdir -p "$NCCL_LOG_DIR"
mkdir -p "$LOG_DIR"

# 序列长度
SEQ_LENGTH=${1:-128}

# 策略列表 (索引:描述:GPU数)
declare -a strategies=(
    "1:单卡基线:1"
    "2:2卡张量并行:2"
    "3:4卡张量并行:4"
    "4:2卡流水线并行:2"
    "5:4卡流水线并行:4"
    "6:2×2混合并行:4"
)

echo "📋 测试配置:"
echo "   序列长度: $SEQ_LENGTH"
echo "   策略数量: ${#strategies[@]}"
echo "   结果目录: $RESULT_DIR"
echo "   NCCL日志: $NCCL_LOG_DIR"
echo ""
echo "======================================================================"

# 统计信息
TOTAL=${#strategies[@]}
SUCCESS=0
FAILED=0

# 遍历策略
for strategy_info in "${strategies[@]}"; do
    IFS=':' read -r idx desc num_gpus <<< "$strategy_info"
    
    echo ""
    echo "######################################################################"
    echo "# 策略 $idx/$TOTAL: $desc"
    echo "######################################################################"
    echo ""
    
    # 日志文件
    NCCL_LOG="$NCCL_LOG_DIR/strategy_${idx}_nccl.log"
    FULL_LOG="$NCCL_LOG_DIR/strategy_${idx}_full.log"
    
    # 设置NCCL环境变量
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=COLL,INIT,ENV
    export RESULT_DIR="$RESULT_DIR"  # 设置结果保存目录
    
    echo "🔍 启用NCCL Profiling..."
    echo "   NCCL_DEBUG=INFO"
    echo "   NCCL_DEBUG_SUBSYS=COLL,INIT,ENV"
    echo "   RESULT_DIR=$RESULT_DIR"
    echo ""
    
    # 运行单个策略测试（独立进程）
    if python -u "$TEST_SCRIPT" "$idx" "$SEQ_LENGTH" \
        -ll:gpu "$num_gpus" \
        -ll:zsize $((20000 * num_gpus)) \
        -ll:fsize 14000 \
        -ll:cpu $((4 * num_gpus)) \
        2>&1 | tee "$FULL_LOG"; then
        
        echo ""
        echo "✅ 策略 $idx 测试成功"
        
        # 提取NCCL通信日志
        echo "📊 提取NCCL通信数据..."
        grep "NCCL INFO" "$FULL_LOG" > "$NCCL_LOG" || echo "   (未找到NCCL INFO日志)"
        
        # 统计通信操作
        if [ -s "$NCCL_LOG" ]; then
            ALLREDUCE_COUNT=$(grep -c "AllReduce" "$NCCL_LOG" || echo "0")
            BROADCAST_COUNT=$(grep -c "Broadcast" "$NCCL_LOG" || echo "0")
            SEND_RECV_COUNT=$(grep -c "Send\|Recv" "$NCCL_LOG" || echo "0")
            
            echo "   AllReduce 操作: $ALLREDUCE_COUNT 次"
            echo "   Broadcast 操作: $BROADCAST_COUNT 次"
            echo "   Send/Recv 操作: $SEND_RECV_COUNT 次"
            
            # 提取通信带宽（如果有）
            if grep -q "busbw" "$NCCL_LOG"; then
                echo "   通信带宽数据:"
                grep "busbw" "$NCCL_LOG" | tail -5
            fi
        else
            echo "   ⚠️  未检测到NCCL通信（可能是单卡或通信日志未启用）"
        fi
        
        SUCCESS=$((SUCCESS + 1))
    else
        EXIT_CODE=$?
        echo ""
        echo "❌ 策略 $idx 测试失败 (exit code: $EXIT_CODE)"
        echo "   完整日志: $FULL_LOG"
        FAILED=$((FAILED + 1))
    fi
    
    # 清理环境变量
    unset NCCL_DEBUG
    unset NCCL_DEBUG_SUBSYS
    unset RESULT_DIR
    
    echo ""
    echo "======================================================================"
    echo "📊 进度: $((SUCCESS + FAILED))/$TOTAL (成功: $SUCCESS, 失败: $FAILED)"
    echo "======================================================================"
    
    # 等待2秒，让系统清理资源
    sleep 2
done

echo ""
echo "======================================================================"
echo "📈 测试完成！"
echo "======================================================================"
echo "总计: $TOTAL"
echo "成功: $SUCCESS ✅"
echo "失败: $FAILED ❌"
echo ""

# 生成NCCL通信分析报告
if [ $SUCCESS -gt 0 ]; then
    echo "📊 分析NCCL通信数据..."
    python "$WORKSPACE/analyze_nccl_logs.py" "$NCCL_LOG_DIR" 2>&1 || echo "⚠️  NCCL分析脚本未找到或执行失败"
    echo ""
fi

echo "📁 结果文件:"
echo "   结果目录: $RESULT_DIR"
echo "   性能数据: $RESULT_DIR/result_*.json"
echo "   NCCL日志: $NCCL_LOG_DIR/strategy_*_nccl.log"
echo "   完整日志: $NCCL_LOG_DIR/strategy_*_full.log"
echo ""
echo "💡 提示:"
echo "   - 旧的测试结果保存在: /workspace/test_results"
echo "   - 新的NCCL profiling结果保存在: $RESULT_DIR"
echo ""

echo "======================================================================"
echo "✅ 所有任务完成！"
echo "======================================================================"
echo ""

# 返回失败数量
exit $FAILED

