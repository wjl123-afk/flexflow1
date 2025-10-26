#!/usr/bin/env python3
"""
NCCL日志分析工具
从NCCL profiling日志中提取真实的通信数据
"""

import os
import sys
import re
import json
from pathlib import Path
from collections import defaultdict


def parse_nccl_log(log_file):
    """解析单个NCCL日志文件"""
    stats = {
        'allreduce_count': 0,
        'broadcast_count': 0,
        'send_count': 0,
        'recv_count': 0,
        'total_comm_ops': 0,
        'bandwidth_samples': [],
        'comm_size_bytes': []
    }
    
    if not os.path.exists(log_file):
        return stats
    
    with open(log_file, 'r') as f:
        for line in f:
            # 统计通信操作类型
            if 'AllReduce' in line:
                stats['allreduce_count'] += 1
            elif 'Broadcast' in line:
                stats['broadcast_count'] += 1
            elif 'Send' in line:
                stats['send_count'] += 1
            elif 'Recv' in line:
                stats['recv_count'] += 1
            
            # 提取通信带宽 (格式: busbw 123.45 GB/s)
            bw_match = re.search(r'busbw\s+([\d.]+)', line)
            if bw_match:
                stats['bandwidth_samples'].append(float(bw_match.group(1)))
            
            # 提取通信数据量 (格式: size 12345 bytes)
            size_match = re.search(r'size\s+(\d+)', line)
            if size_match:
                stats['comm_size_bytes'].append(int(size_match.group(1)))
    
    stats['total_comm_ops'] = (stats['allreduce_count'] + 
                                stats['broadcast_count'] + 
                                stats['send_count'] + 
                                stats['recv_count'])
    
    # 计算平均带宽
    if stats['bandwidth_samples']:
        stats['avg_bandwidth_gbps'] = sum(stats['bandwidth_samples']) / len(stats['bandwidth_samples'])
    else:
        stats['avg_bandwidth_gbps'] = 0.0
    
    # 计算总通信量
    if stats['comm_size_bytes']:
        stats['total_comm_gb'] = sum(stats['comm_size_bytes']) / (1024**3)
        stats['avg_comm_size_mb'] = (sum(stats['comm_size_bytes']) / len(stats['comm_size_bytes'])) / (1024**2)
    else:
        stats['total_comm_gb'] = 0.0
        stats['avg_comm_size_mb'] = 0.0
    
    return stats


def calculate_comm_overhead(result_file, nccl_stats):
    """
    根据性能数据和NCCL统计计算真实通信开销
    
    假设：
    - 单卡延迟 = 纯计算时间 (无通信)
    - 多卡延迟 = 计算时间 + 通信时间
    - 通信开销% = (多卡延迟 - 单卡延迟) / 多卡延迟 * 100
    """
    if not os.path.exists(result_file):
        return None
    
    with open(result_file, 'r') as f:
        result = json.load(f)
    
    e2e_latency = result['metrics']['e2e_latency_ms']
    
    # 假设单卡延迟为基线（纯计算）
    # 这个值应该从 result_TP1_PP1_SL128.json 读取
    baseline_latency = 1702.97  # 单卡基线延迟
    
    if e2e_latency > baseline_latency:
        overhead_ms = e2e_latency - baseline_latency
        overhead_percent = (overhead_ms / e2e_latency) * 100
    else:
        overhead_ms = 0
        overhead_percent = 0
    
    return {
        'e2e_latency_ms': e2e_latency,
        'baseline_latency_ms': baseline_latency,
        'overhead_ms': overhead_ms,
        'overhead_percent': overhead_percent,
        'total_comm_ops': nccl_stats['total_comm_ops'],
        'allreduce_count': nccl_stats['allreduce_count'],
        'avg_bandwidth_gbps': nccl_stats['avg_bandwidth_gbps'],
        'total_comm_gb': nccl_stats['total_comm_gb']
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_nccl_logs.py <nccl_log_dir>")
        sys.exit(1)
    
    nccl_log_dir = Path(sys.argv[1])
    result_dir = nccl_log_dir.parent
    
    print("\n" + "="*70)
    print("📊 NCCL通信数据分析报告")
    print("="*70 + "\n")
    
    # 策略名称映射
    strategy_names = {
        1: "TP1_PP1 (单卡基线)",
        2: "TP2_PP1 (2卡张量并行)",
        3: "TP4_PP1 (4卡张量并行)",
        4: "TP1_PP2 (2卡流水线并行)",
        5: "TP1_PP4 (4卡流水线并行)",
        6: "TP2_PP2 (2×2混合并行)"
    }
    
    # 读取单卡基线
    baseline_file = result_dir / "result_TP1_PP1_SL128.json"
    baseline_latency = 1702.97  # 默认值
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline_data = json.load(f)
            baseline_latency = baseline_data['metrics']['e2e_latency_ms']
        print(f"✅ 单卡基线延迟: {baseline_latency:.2f} ms\n")
    else:
        print(f"⚠️  未找到单卡基线文件，使用默认值: {baseline_latency:.2f} ms\n")
    
    # 分析所有策略
    all_results = []
    
    for idx in range(1, 7):
        nccl_log = nccl_log_dir / f"strategy_{idx}_nccl.log"
        result_file = result_dir / f"result_TP{2 if idx in [2,6] else (4 if idx==3 else 1)}_PP{2 if idx in [4,6] else (4 if idx==5 else 1)}_SL128.json"
        
        strategy_name = strategy_names.get(idx, f"策略{idx}")
        
        print(f"{'─'*70}")
        print(f"策略 {idx}: {strategy_name}")
        print(f"{'─'*70}")
        
        # 解析NCCL日志
        nccl_stats = parse_nccl_log(nccl_log)
        
        if nccl_stats['total_comm_ops'] > 0:
            print(f"📡 NCCL通信统计:")
            print(f"   - AllReduce 操作: {nccl_stats['allreduce_count']} 次")
            print(f"   - Broadcast 操作: {nccl_stats['broadcast_count']} 次")
            print(f"   - Send/Recv 操作: {nccl_stats['send_count'] + nccl_stats['recv_count']} 次")
            print(f"   - 总通信操作: {nccl_stats['total_comm_ops']} 次")
            
            if nccl_stats['avg_bandwidth_gbps'] > 0:
                print(f"   - 平均带宽: {nccl_stats['avg_bandwidth_gbps']:.2f} GB/s")
            
            if nccl_stats['total_comm_gb'] > 0:
                print(f"   - 总通信量: {nccl_stats['total_comm_gb']:.4f} GB")
                print(f"   - 平均通信大小: {nccl_stats['avg_comm_size_mb']:.2f} MB")
        else:
            print(f"   ℹ️  无NCCL通信（单卡或未检测到通信日志）")
        
        # 计算通信开销
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            e2e_latency = result['metrics']['e2e_latency_ms']
            
            # 计算真实通信开销
            if e2e_latency > baseline_latency:
                overhead_ms = e2e_latency - baseline_latency
                overhead_percent = (overhead_ms / e2e_latency) * 100
            else:
                overhead_ms = 0
                overhead_percent = 0
            
            print(f"\n⏱️  性能指标:")
            print(f"   - E2E延迟: {e2e_latency:.2f} ms")
            print(f"   - 基线延迟: {baseline_latency:.2f} ms")
            print(f"   - 开销时间: {overhead_ms:.2f} ms")
            print(f"   - 开销百分比: {overhead_percent:.1f}% (真实测量)")
            
            # 保存结果
            all_results.append({
                'strategy_id': idx,
                'strategy_name': strategy_name,
                'e2e_latency_ms': e2e_latency,
                'overhead_ms': overhead_ms,
                'overhead_percent': overhead_percent,
                'nccl_stats': nccl_stats
            })
        else:
            print(f"\n⚠️  未找到性能数据文件: {result_file.name}")
        
        print()
    
    # 生成汇总对比
    print("="*70)
    print("📈 通信开销对比汇总")
    print("="*70 + "\n")
    
    print(f"{'策略':<25} {'延迟(ms)':<12} {'开销(ms)':<12} {'开销%':<10} {'通信操作':<10}")
    print("─"*70)
    
    for result in all_results:
        print(f"{result['strategy_name']:<25} "
              f"{result['e2e_latency_ms']:<12.2f} "
              f"{result['overhead_ms']:<12.2f} "
              f"{result['overhead_percent']:<10.1f} "
              f"{result['nccl_stats']['total_comm_ops']:<10}")
    
    print()
    
    # 保存详细报告
    report_file = result_dir / "nccl_analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            'baseline_latency_ms': baseline_latency,
            'strategies': all_results
        }, f, indent=2)
    
    print(f"💾 详细报告已保存: {report_file}")
    print()
    
    # 生成CSV
    csv_file = result_dir / "nccl_comm_overhead.csv"
    with open(csv_file, 'w') as f:
        f.write("Strategy,TP,PP,GPUs,E2E_Latency_ms,Overhead_ms,Overhead_percent,AllReduce_ops,Total_comm_ops\n")
        
        tp_pp_map = {
            1: (1, 1, 1),
            2: (2, 1, 2),
            3: (4, 1, 4),
            4: (1, 2, 2),
            5: (1, 4, 4),
            6: (2, 2, 4)
        }
        
        for result in all_results:
            idx = result['strategy_id']
            tp, pp, gpus = tp_pp_map.get(idx, (1, 1, 1))
            strategy_short = f"TP{tp}_PP{pp}"
            
            f.write(f"{strategy_short},{tp},{pp},{gpus},"
                   f"{result['e2e_latency_ms']:.2f},"
                   f"{result['overhead_ms']:.2f},"
                   f"{result['overhead_percent']:.1f},"
                   f"{result['nccl_stats']['allreduce_count']},"
                   f"{result['nccl_stats']['total_comm_ops']}\n")
    
    print(f"📊 CSV报告已保存: {csv_file}")
    print()
    
    print("="*70)
    print("✅ NCCL分析完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

