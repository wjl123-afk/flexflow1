#!/bin/bash
# FlexFlow并行策略测试 - 主控脚本
# 为每个策略启动独立的Python进程，避免单例冲突

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "🚀 FlexFlow 并行策略性能测试套件"
echo "======================================================================"
echo ""

# 配置变量
WORKSPACE="/workspace"
TEST_SCRIPT="$WORKSPACE/test_parallel_strategies.py"
SUMMARY_FILE="$WORKSPACE/profiling_summary.json"
LOG_DIR="$WORKSPACE/profiling_logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 策略列表
declare -a strategies=(
    "1:单卡基线:1"
    "2:2卡张量并行:2"
    "3:4卡张量并行:4"
    "4:2卡流水线并行:2"
    "5:4卡流水线并行:4"
    "6:2×2混合并行:4"
)

# 序列长度
SEQ_LENGTH=${1:-128}

echo "📋 测试配置:"
echo "   序列长度: $SEQ_LENGTH"
echo "   策略数量: ${#strategies[@]}"
echo "   日志目录: $LOG_DIR"
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
    
    LOG_FILE="$LOG_DIR/strategy_${idx}_$(date +%Y%m%d_%H%M%S).log"
    
    # 运行单个策略测试（独立进程）
    if python -u "$TEST_SCRIPT" "$idx" "$SEQ_LENGTH" \
        -ll:gpu "$num_gpus" \
        -ll:zsize $((20000 * num_gpus)) \
        -ll:fsize 14000 \
        -ll:cpu $((4 * num_gpus)) \
        2>&1 | tee "$LOG_FILE"; then
        
        echo ""
        echo "✅ 策略 $idx 测试成功"
        SUCCESS=$((SUCCESS + 1))
    else
        EXIT_CODE=$?
        echo ""
        echo "❌ 策略 $idx 测试失败 (exit code: $EXIT_CODE)"
        echo "   日志文件: $LOG_FILE"
        FAILED=$((FAILED + 1))
    fi
    
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

# 生成汇总报告
if [ $SUCCESS -gt 0 ]; then
    echo "📊 生成汇总报告..."
    python "$WORKSPACE/analyze_profiling_results.py" 2>&1 || echo "⚠️  分析脚本执行失败（可能缺少结果文件）"
    echo ""
    
    if [ -f "$SUMMARY_FILE" ]; then
        echo "💾 汇总文件: $SUMMARY_FILE"
    fi
    
    echo "📁 结果文件:"
    ls -lh "$WORKSPACE"/result_*.json 2>/dev/null || echo "   (未找到结果文件)"
    echo ""
fi

echo "======================================================================"
echo "✅ 所有任务完成！"
echo "======================================================================"
echo ""

# 返回成功数量
exit $FAILED

